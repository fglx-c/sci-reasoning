# Generative Reward Model Integration Guide

This README documents every change applied to **simpleRL-reason** to replace the original rule-based reward functions with the LLM-based *General-Verifier* reward model borrowed from **General-Reasoner**.

---
## 1. Overview
* **Goal** – Use an LLM verifier to score responses instead of hard-coded math functions.
* **Key result** – Rewards are now supplied exclusively by the new *verifier worker*; function-based logic is bypassed.

---
## 2. New File
| Path | Purpose |
|------|---------|
| `simpleRL-reason/verifier.py` | Worker that loads the General-Verifier model via `vllm` and computes `rm_scores` for each batch. |

---
## 3. Modified Files
| File | What changed |
|------|--------------|
| `verl/trainer/main_ppo.py` | Added `elif config.reward_model.strategy == 'verifier'` branch to import the new `RewardModelWorker`. |
| `verl/trainer/config/ppo_trainer.yaml` | • `reward_model.enable = True`  
  • `reward_model.strategy = verifier`  
  • `reward_model.model.path` set to **`/home/ec2-user/dataset/training/models/general-verifier`** (downloaded locally).  
  • Older FSDP parameters retained but ignored by the verifier. |
| `start_grpo_ray_job_nvidia.sh` | • `TRAIN_FILE`/`VAL_FILE` now point to General-Reasoner parquet files.  
  • Checkpoint folder & experiment name renamed to `general_reasoner`. |

---
## 4. Downloaded Assets
```bash
huggingface-cli download TIGER-Lab/general-verifier \
    --local-dir /home/ec2-user/dataset/training/models/general-verifier \
    --local-dir-use-symlinks False
```
This places the verifier weights where the YAML expects them.

---
## 5. Dataset Location
By default the launch script now uses the original SimpleRL-Reason math dataset:
```
/home/ec2-user/dataset/simpleRL-reason/simplelr_math_35/train.parquet
/home/ec2-user/dataset/simpleRL-reason/simplelr_math_35/test.parquet
```
If you prefer any other parquet, adjust *TRAIN_FILE* / *VAL_FILE* in `start_grpo_ray_job_nvidia.sh` accordingly.

---
## 6. How to Run
```bash
cd /home/ec2-user/dataset/simpleRL-reason
bash start_grpo_ray_job_nvidia.sh \
  --model_name General-Reasoner-Backbone-Name \
  --max_response_length 2048 \
  --train_batch_size 512 \
  # (plus any other overrides you need)
```
The script will:
1. Verify conda env and Ray cluster.
2. Submit `python -m verl.trainer.main_ppo` with many CLI overrides.
3. Training logs & checkpoints go to the updated `CHECKPOINT_DIR`.

---
## 7. Internal Logic Flow
1. `main_ppo.py` reads `ppo_trainer.yaml` (Hydra).  
2. `reward_model.strategy = verifier` triggers import of `verifier.RewardModelWorker`.  
3. Ray launches that worker; it loads *General-Verifier* via *vllm*.  
4. During PPO rollout `rm_scores` are returned; `RewardManager` detects them and skips any function-based scoring.  
5. PPO proceeds with these scores to compute advantages.

---
## 8. Troubleshooting
| Issue | Fix |
|-------|-----|
| *FileNotFoundError* for parquet files | Verify paths or update script variables. |
| *CUDA OOM* in verifier | Reduce `rollout.gpu_memory_utilization` or use a smaller verifier model. |
| *ImportError vllm* | Ensure `vllm` is installed in the active conda env. |

---
## 9. Future Work
* Support batching larger than 1024 tokens if needed (edit `sequence_strs.append(...)` truncation).
* Parameter-offload options for verifier when GPU memory is tight.

---
**Last updated:** $(date) 