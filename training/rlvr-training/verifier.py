import logging
import os
import re

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl import DataProto
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.
    Returns:
        The content inside the last \boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.
    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]

    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()

    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)


class RewardModelWorker(Worker):
    """LLM-based verifier reward worker."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Load verifier model and tokenizer."""
        # The default code tries to shard the verifier across WORLD_SIZE GPUs
        # and launches additional Ray actors.  That over-subscribes GPU
        # resources when the training job already occupies all devices.

        # Instead we force the verifier to run *locally* inside the current
        # process and on the single GPU Ray assigned to it.  This removes the
        # need for extra GPU actors and lets the full pipeline fit on a
        # 4-GPU node.

        import os
        self.llm = LLM(
            model=self.config.model.path,
            tensor_parallel_size=1,     # single-GPU verifier
            gpu_memory_utilization=0.5,
            dtype=torch.bfloat16,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """Compute reward scores for a batch using the verifier."""
        torch.cuda.empty_cache()

        sequence_strs, ground_truths, questions, valid_response_lengths = [], [], [], []

        # Build strings and metadata for each sample
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(item.batch["attention_mask"][:prompt_len].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_resp_len = int(item.batch["attention_mask"][prompt_len:].sum())
            valid_response_lengths.append(valid_resp_len)

            seq = torch.cat((valid_prompt_ids, response_ids[:valid_resp_len]))
            # Truncate to last 1024 tokens to control verifier context length
            sequence_strs.append(self.tokenizer.decode(seq[-1024:]))

            questions.append(item.non_tensor_batch["extra_info"]["question"])
            ground_truths.append(item.non_tensor_batch["reward_model"]["ground_truth"])

        # Extract solutions produced by policy
        solutions = [extract_solution(s) for s in sequence_strs]

        # Compose verifier prompts
        messages = [
            VERIFIER_PROMPT_TEMPLATE.format(question=q, ground_truth=gt, student_answer=sol)
            for q, gt, sol in zip(questions, ground_truths, solutions)
        ]

        # Run verifier LLM
        outputs = self.llm.generate(messages, self.sampling_params)
        verifications = [o.outputs[0].text.strip() for o in outputs]

        # Fill reward tensor (token-level, score at last response token)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i, (gt, sol, verify, resp_len) in enumerate(
            zip(ground_truths, solutions, verifications, valid_response_lengths)
        ):
            score = -0.5 if sol is None else 0.0
            if not sol:
                sol = "No Answer"
            if VERIFIER_PASS_TAG in verify:
                score += 1.0
                diff = abs(len(self.tokenizer.encode(sol)) - len(self.tokenizer.encode(gt)))
                score -= min(diff, 10) * 0.05
            reward_tensor[i, resp_len - 1] = score

        batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        torch.cuda.empty_cache()
        return DataProto(batch=batch) 