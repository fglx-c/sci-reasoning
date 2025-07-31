import logging
import os
import re
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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

PAIRWISE_COMPARISON_TEMPLATE = (
    "Question: {question}\n\n"
    "Ground Truth Answer: {ground_truth}\n\n"
    "Response A: {response_a}\n\n"
    "Response B: {response_b}\n\n"
    "Compare these responses considering accuracy, reasoning quality, and clarity.\n"
    "Which response is better?\n\n"
    "Answer: A or B"
)


def extract_last_boxed(text: str) -> str:
    """Extract the last occurrence of a boxed answer from the input text."""
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """Try to extract the final answer from the text using several candidate patterns."""
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

    if last_match:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        for stop_word in stop_words:
            if last_match.endswith(stop_word):
                last_match = last_match[:-len(stop_word)].strip()

    return last_match


def extract_solution(solution_str: str) -> str:
    """Extract solution from response text."""
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)


class PairwiseRewardModelWorker(Worker):
    """Pairwise ranking-based reward worker."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sampling_params = SamplingParams(
            temperature=config.pairwise_config.get('comparison_temperature', 0.0), 
            max_tokens=config.pairwise_config.get('max_tokens', 10)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Load comparison model and tokenizer."""
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
        """Compute pairwise ranking-based reward scores using binary search insertion."""
        torch.cuda.empty_cache()

        # Group samples by prompt UID for pairwise ranking
        groups = self._group_by_prompt(data)
        
        # Build reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        for uid, samples in groups.items():
            if len(samples) == 1:
                # Single sample gets neutral reward
                sample = samples[0]
                reward_tensor[sample['index'], sample['valid_response_length'] - 1] = 0.0
                continue
            
            # Extract metadata (same for all samples in group)
            question = samples[0]['question']
            ground_truth = samples[0]['ground_truth']
            
            # Build ranking via binary search with pairwise comparisons
            ranking = self._build_ranking(samples, question, ground_truth)
            
            # Convert ranking to linear rewards
            rewards = self._ranking_to_rewards(ranking)
            
            # Assign rewards to tensor
            for sample_idx, reward in rewards:
                sample = next(s for s in samples if s['index'] == sample_idx)
                reward_tensor[sample_idx, sample['valid_response_length'] - 1] = reward

        batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        torch.cuda.empty_cache()
        return DataProto(batch=batch)
    
    def _group_by_prompt(self, data: DataProto) -> Dict[str, List[Dict]]:
        """Group samples by prompt - using question as UID since actual UID might not exist."""
        groups = defaultdict(list)
        
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(item.batch["attention_mask"][:prompt_len].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_resp_len = int(item.batch["attention_mask"][prompt_len:].sum())

            # Decode response text - Miaosen Chai
            seq = torch.cat((valid_prompt_ids, response_ids[:valid_resp_len]))
            sequence_str = self.tokenizer.decode(seq[-1024:])  # Truncate to last 1024 tokens
            
            # Extract solution from response
            solution = extract_solution(sequence_str)
            if not solution:
                solution = "No Answer"

            # Get metadata
            question = item.non_tensor_batch["extra_info"]["question"]
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            
            # Use question as UID for grouping (since samples from same prompt have same question)
            uid = question

            sample_info = {
                'index': i,
                'sequence_str': sequence_str,
                'solution': solution,
                'question': question,
                'ground_truth': ground_truth,
                'valid_response_length': valid_resp_len
            }
            
            groups[uid].append(sample_info)
        
        return groups
    
    def _build_ranking(self, samples: List[Dict], question: str, ground_truth: str) -> List[Dict]:
        """Build ranking via binary search insertion with pairwise comparisons."""
        # Start with random sample
        ranking = [random.choice(samples)]
        remaining = [s for s in samples if s['index'] != ranking[0]['index']]
        
        # Insert each remaining sample using binary search
        for sample in remaining:
            position = self._binary_search_insert(ranking, sample, question, ground_truth)
            ranking.insert(position, sample)
        
        return ranking
    
    def _binary_search_insert(self, ranking: List[Dict], new_sample: Dict, question: str, ground_truth: str) -> int:
        """Find insertion position using pairwise comparisons."""
        left, right = 0, len(ranking)
        
        while left < right:
            mid = (left + right) // 2
            mid_sample = ranking[mid]
            
            # Compare samples directly
            comparison = self._compare_samples(mid_sample, new_sample, question, ground_truth)
            
            if comparison == "A":  # mid_sample is better
                left = mid + 1
            else:  # new_sample is better (B)
                right = mid
        
        return left
    
    def _compare_samples(self, sample_a: Dict, sample_b: Dict, question: str, ground_truth: str) -> str:
        """Perform pairwise comparison with retry logic."""
        retry_on_invalid = self.config.pairwise_config.get('retry_on_invalid', True)
        max_retries = 3 if retry_on_invalid else 1
        
        for attempt in range(max_retries):
            # Build comparison prompt
            prompt = PAIRWISE_COMPARISON_TEMPLATE.format(
                question=question,
                ground_truth=ground_truth,
                response_a=sample_a['solution'],
                response_b=sample_b['solution']
            )
            
            # Generate comparison
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # Parse response
            if "A" in response.upper() and "B" not in response.upper():
                return "A"
            elif "B" in response.upper() and "A" not in response.upper():
                return "B"
            elif attempt == max_retries - 1:
                # Fallback: random choice if all attempts fail
                return random.choice(["A", "B"])
        
        return random.choice(["A", "B"])
    
    def _ranking_to_rewards(self, ranking: List[Dict]) -> List[Tuple[int, float]]:
        """Convert ranking positions to linear rewards."""
        n = len(ranking)
        if n == 1:
            return [(ranking[0]['index'], 0.0)]
        
        # Linear distribution: best=1.0, worst=0.0
        rewards = []
        for i, sample in enumerate(ranking):
            reward = (n - 1 - i) / (n - 1)  # Higher rank = higher reward
            rewards.append((sample['index'], reward))
        
        return rewards 