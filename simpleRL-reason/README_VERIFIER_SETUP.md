# GRPO Training Setup & Troubleshooting Guide

This document summarizes all the solutions implemented to get the GRPO (Group Relative Policy Optimization) training working properly with the OpenReasoning-Nemotron-1.5B model.

## 🚀 Quick Start

To run the working GRPO training:

```bash
# 1. Activate environment
conda activate mat-verl-py311

# 2. Navigate to training directory
cd /home/ec2-user/dataset/simpleRL-reason

# 3. Run the small test training
bash start_grpo_ray_job_nvidia_small_test.sh
```

## 🔧 Major Issues Fixed

### 1. **Disk Space Expansion**

**Problem**: Root filesystem was 100% full (200G used, 0 available)  
**Solution**: Expanded EBS volume and filesystem to use additional space

```bash
# Expand partition to use all available disk space
sudo growpart /dev/nvme0n1 1

# Extend XFS filesystem to use the new partition space
sudo xfs_growfs /

# Result: Gained 30G additional space (200G → 230G)
```

### 2. **Empty Dataloader Issue**

**Problem**: `AssertionError: assert len(self.train_dataloader) >= 1`  
**Root Cause**: Dataset had fewer samples than batch size, causing entire dataset to be dropped with `drop_last=True`

**Solution**: Created dataset with appropriate number of samples matching batch size requirements

```python
# Fixed dataset creation with minimum required samples
train_small_size = 12  # Ensures compatibility with batch_size and GPU constraints
```

### 3. **GPU Configuration & Batch Size Constraints**

**Problem**: `AssertionError: real_train_batch_size (X) must be divisible by total n_gpus (4)`  
**Root Cause**: Mathematical constraint not satisfied for distributed training

**Solution**: Aligned batch size, dataset size, and GPU configuration:

```bash
# Final working configuration
TRAIN_BATCH_SIZE=12      # Divisible by 4 GPUs (validation) and 3 workers (chunking)
trainer.n_gpus_per_node=4  # Use all 4 available GPUs
```

**Key Insight**: Despite configuring 4 GPUs, the system allocates:
- 3 GPUs for data parallel training workers
- 1 GPU for reward model/verifier components

### 4. **Data Chunking Issues**

**Problem**: `AssertionError: only support equal chunk. Got size of DataProto X and chunk Y`  
**Root Cause**: Dataset size not evenly divisible by number of actual workers

**Solution**: Used batch_size=12 which satisfies both constraints:
- `12 % 4 == 0` ✅ (passes GPU validation)  
- `12 ÷ 3 = 4` samples per worker ✅ (perfect chunking)

### 5. **PPO Mini-Batch Configuration**

**Problem**: `RuntimeError: split_size must be a positive integer, but got 0`  
**Root Cause**: Mini-batch size too small for the number of samples per worker

**Solution**: Adjusted mini-batch sizes to match worker allocation:

```bash
# Each worker gets 4 samples (12÷3=4), so set mini-batch to 4
actor_rollout_ref.actor.ppo_mini_batch_size=4
critic.ppo_mini_batch_size=4
```

### 6. **Memory Optimization**

**Problem**: Out of Memory (OOM) errors during training  
**Solution**: Applied multiple memory optimization strategies:

```bash
# Reduced GPU memory utilization
ROLLOUT_GPU_MEMORY_UTIL=0.3  # Reduced from 0.5

# Enabled FSDP offloading to reduce GPU memory usage
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.grad_offload=True  
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

## 📊 Working Configuration

### Final Parameters
```bash
MODEL_NAME="OpenReasoning-Nemotron-1.5B"
MAX_RESPONSE_LENGTH=512
TRAIN_BATCH_SIZE=12
ROLLOUT_N=1
trainer.n_gpus_per_node=4
trainer.total_epochs=2
```

### GPU Allocation
- **4 GPUs total** configured in Ray cluster
- **3 GPUs** for data parallel training workers  
- **1 GPU** for reward model/verifier components
- **Memory utilization**: 30% per GPU (optimized for stability)

### Dataset Configuration
- **12 training samples** (creates 1 dataloader batch of size 12)
- **1 validation sample**
- **Batch processing**: 4 samples per worker across 3 workers

## 🔍 Training Validation

### Successful Training Indicators
✅ **Configuration validation passed**  
✅ **All 4 GPUs utilized efficiently**  
✅ **Ray job completes with exit code 0**  
✅ **Model checkpoints saved successfully**  
✅ **Both training steps completed** (`step:1` and `step:2`)

### Key Log Messages to Look For
```
[validate_config] All configuration checks passed successfully!
Size of train dataloader: 1
Total training steps: 2
Job 'raysubmit_XXXXXXXX' succeeded
```

### Checkpoint Location
```
/home/ec2-user/dataset/training/checkpoints/verl-grpo_OpenReasoning-Nemotron-1.5B_SMALL_TEST_max_response512_batch12_rollout1_klcoef0.0001_entcoef0.001/
```

## 🛠️ Environment Requirements

### Conda Environment
```bash
conda activate mat-verl-py311
```

### Required Models
- **Actor/Policy Model**: `/home/ec2-user/dataset/training/models/OpenReasoning-Nemotron-1.5B`
- **Reward Model**: `/home/ec2-user/dataset/training/models/general-verifier`

### Ray Cluster Configuration
```bash
# Ray starts automatically with the script
ray start --head --node-ip-address 172.31.31.243 --num-gpus 4
```

## 📈 Performance Metrics

### Training Timing (per step)
- **Generation**: ~39s
- **Log probability calculation**: ~8s  
- **Reference model**: ~7s
- **Advantage estimation**: ~0.5s
- **Actor update**: ~21s
- **Validation**: ~20s
- **Checkpoint saving**: ~48s
- **Total step time**: ~143s

### Resource Utilization
- **GPU Memory**: ~30% per GPU (optimized)
- **System RAM**: Up to 98% during training (normal)
- **Model Parameters**: 1.54B (Qwen2 architecture)

## 🔄 Ray Job Management

### Useful Commands
```bash
# Check Ray cluster status
ray status

# List active jobs
ray job list

# View job logs
ray job logs <job-id>

# Stop a job
ray job stop <job-id>

# Access Ray Dashboard
# http://127.0.0.1:8265
```

## 📝 Training Script Usage

### Script Location
```
/home/ec2-user/dataset/simpleRL-reason/start_grpo_ray_job_nvidia_small_test.sh
```

### Key Features
- **Automatic Ray cluster management** (start/stop/restart)
- **Configuration validation** before training
- **Memory optimization** settings
- **Comprehensive logging** with timing metrics
- **Automatic checkpoint saving** every step

### Expected Runtime
- **Small test (2 epochs, 12 samples)**: ~5-10 minutes
- **Memory usage**: Peaks at ~98% system RAM (normal)
- **GPU utilization**: All 4 GPUs active throughout training

## 🏗️ System Architecture

### Model Components
1. **Actor Model** (Policy): Generates responses during rollout
2. **Reference Model**: Provides baseline for KL divergence calculation  
3. **Critic Model**: Estimates value functions for advantage calculation
4. **Reward Model**: Scores generated responses for training signal

### Data Flow
1. **Rollout**: Generate responses using Actor model
2. **Scoring**: Evaluate responses with Reward model  
3. **Advantage**: Calculate advantages using Critic model
4. **Update**: Train Actor and Critic with PPO objectives

## 🎯 Success Metrics

The training setup is working correctly when:
- Ray job completes successfully (exit code 0)
- All GPU workers initialize without errors
- Dataloader has correct size (matches batch configuration)
- Model checkpoints are saved after each step
- Training completes both epochs without OOM errors

This configuration provides a stable foundation for GRPO training experiments. 