#!/bin/bash

# GRPO Training Script for SMALL TEST DATASET using Ray Job Submit
# This script is designed for quick testing with minimal data (1 sample)

set -e  # Exit on any error

USER_ENV=`whoami`
echo "Starting GRPO SMALL TEST training with Ray Job Submit for user: $USER_ENV"

# Environment setup
export PYTHONUNBUFFERED=1
# Note: Different Ray commands need different address formats
export RAY_ADDRESS="172.31.31.243:6379"  # For ray status
export RAY_JOB_ADDRESS="http://127.0.0.1:8265"  # For ray job submit (dashboard is localhost-only)
# Prevent NCCL from hanging on systems without EFA/OFI by disabling the OFI plugin
export NCCL_NET_OFI_DISABLE=1

# Ensure a clean Ray runtime by force-stopping any leftovers from previous runs
echo "Stopping any existing Ray cluster (if running)..."
ray stop --force || true

# SMALL TEST PARAMETERS - Optimized for minimal testing
MODEL_NAME="OpenReasoning-Nemotron-1.5B"
MAX_RESPONSE_LENGTH=512  # Reduced from 8192
TRAIN_BATCH_SIZE=12      # Divisible by 4 GPUs (validation) and 3 workers (chunking)
ROLLOUT_N=1              # 1 rollout per sample for minimal testing
KL_LOSS_COEF=0.0001
ENTROPY_COEF=0.001
ROLLOUT_GPU_MEMORY_UTIL=0.3  # Further reduced to prevent OOM
ROLLOUT_TP=1
SAVE_FREQ=1              # Save after every epoch
TOTAL_EPOCHS=2           # Only 2 epochs for quick test
REWARD_STRATEGY="pairwise"  # Default strategy: pairwise or verifier
PAIRWISE_TEMP=0.0        # Temperature for pairwise comparisons
PAIRWISE_MAX_TOKENS=10   # Max tokens for pairwise comparison responses
PAIRWISE_CACHE_SIZE=1000 # Cache size for pairwise comparisons

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
        --entropy_coeffient) ENTROPY_COEF="$2"; shift 2 ;;
        --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
        --rollout_tp) ROLLOUT_TP="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --reward_strategy) REWARD_STRATEGY="$2"; shift 2 ;;
        --pairwise_temp) PAIRWISE_TEMP="$2"; shift 2 ;;
        --pairwise_max_tokens) PAIRWISE_MAX_TOKENS="$2"; shift 2 ;;
        --pairwise_cache_size) PAIRWISE_CACHE_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Automatic adjustments for pairwise strategy
if [[ "$REWARD_STRATEGY" == "pairwise" && "$ROLLOUT_N" == "1" ]]; then
    echo "⚠️  Pairwise strategy requires multiple samples for comparison. Setting rollout_n=2"
    ROLLOUT_N=2
fi

# Paths - USING SMALL TEST DATASET
MODEL_PATH="/home/ec2-user/dataset/training/models/$MODEL_NAME"
TRAIN_FILE="/home/ec2-user/dataset/simpleRL-reason/simplelr_math_35_small/train.parquet"  # SMALL DATASET
VAL_FILE="/home/ec2-user/dataset/simpleRL-reason/simplelr_math_35_small/test.parquet"    # SMALL DATASET
WORKING_DIR="/home/ec2-user/dataset/simpleRL-reason"

# Update checkpoint directory name for small test
CHECKPOINT_DIR="/home/ec2-user/dataset/training/checkpoints/verl-grpo_${MODEL_NAME}_SMALL_TEST_max_response${MAX_RESPONSE_LENGTH}_batch${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_klcoef${KL_LOSS_COEF}_entcoef${ENTROPY_COEF}"

echo "==========================================="
echo "GRPO SMALL TEST Training Configuration:"
echo "Model: $MODEL_NAME"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Rollout N: $ROLLOUT_N"
echo "KL Loss Coef: $KL_LOSS_COEF"
echo "Entropy Coef: $ENTROPY_COEF"
echo "GPU Memory Util: $ROLLOUT_GPU_MEMORY_UTIL"
echo "Tensor Parallel: $ROLLOUT_TP"
echo "Save Frequency: $SAVE_FREQ"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Reward Strategy: $REWARD_STRATEGY"
echo "Pairwise Temp: $PAIRWISE_TEMP"
echo "Pairwise Max Tokens: $PAIRWISE_MAX_TOKENS"
echo "Pairwise Cache Size: $PAIRWISE_CACHE_SIZE"
echo "Ray Address: $RAY_JOB_ADDRESS"
echo "Working Dir: $WORKING_DIR"
echo "Train File: $TRAIN_FILE"
echo "Val File: $VAL_FILE"
echo "==========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "mat-verl-py311" ]]; then
    echo "Activating mat-verl-py311 environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate mat-verl-py311
fi

mkdir -p /home/ec2-user/dataset/training/models
mkdir -p /home/ec2-user/dataset/training/checkpoints  
mkdir -p /home/ec2-user/dataset/training/logs

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found at $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Validation file not found at $VAL_FILE"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Check Ray status and dashboard
echo "Checking Ray cluster and dashboard..."
export RAY_ADDRESS="172.31.31.243:6379"
if ! ray status > /dev/null 2>&1; then
    echo "Ray cluster not running. Starting Ray..."
    ray start --head --node-ip-address 172.31.31.243 --num-gpus 4
    sleep 10
    echo "Waiting for dashboard to be ready..."
    
    # Wait for dashboard to be accessible
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8265/api/jobs > /dev/null 2>&1; then
            echo "Dashboard is ready!"
            break
        fi
        echo "Waiting for dashboard... ($i/30)"
        sleep 2
    done
else
    echo "Ray cluster is already running."
    
    # Check if dashboard is accessible
    if ! curl -s http://127.0.0.1:8265/api/jobs > /dev/null 2>&1; then
        echo "Dashboard not accessible. Please restart Ray cluster."
        ray stop --force
        ray start --head --node-ip-address 172.31.31.243 --num-gpus 4
        sleep 10
    fi
fi

# Verify Ray dashboard is accessible
if ! curl -s http://127.0.0.1:8265/api/jobs > /dev/null 2>&1; then
    echo "Error: Ray dashboard is not accessible at $RAY_JOB_ADDRESS"
    echo "Please check Ray cluster status manually."
    exit 1
fi

echo "Starting GRPO SMALL TEST training with Ray Job Submit..."

# Calculate max tokens (reduced for test)
max_num_batched_tokens=$((1024 + $MAX_RESPONSE_LENGTH + 200))

# Submit training job to Ray
export RAY_ADDRESS="$RAY_JOB_ADDRESS"
ray job submit \
    --entrypoint-num-cpus=1 \
    --runtime-env-json='{
        "working_dir": "'$WORKING_DIR'",
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
            "NCCL_NET_OFI_DISABLE": "1"
        }
    }' \
    -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.micro_rollout_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.enable=True \
    reward_model.strategy=$REWARD_STRATEGY \
    reward_model.model.path=/home/ec2-user/dataset/training/models/general-verifier \
    reward_model.pairwise_config.comparison_temperature=$PAIRWISE_TEMP \
    reward_model.pairwise_config.max_tokens=$PAIRWISE_MAX_TOKENS \
    reward_model.pairwise_config.cache_size=$PAIRWISE_CACHE_SIZE \
    reward_model.pairwise_config.retry_on_invalid=true \
    reward_model.micro_batch_size_per_gpu=1 \
    critic.model.path=/home/ec2-user/dataset/training/models/OpenReasoning-Nemotron-1.5B \
    critic.ppo_mini_batch_size=4 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=verl_train_small_test \
    trainer.remove_previous_ckpt=False \
    trainer.experiment_name="verl-grpo_${MODEL_NAME}_SMALL_TEST_max_response${MAX_RESPONSE_LENGTH}_batch${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_klcoef${KL_LOSS_COEF}_entcoef${ENTROPY_COEF}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.remove_clip=False \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=1 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.total_epochs=$TOTAL_EPOCHS

echo ""
echo "SMALL TEST Job submitted to Ray cluster!"
echo "You can monitor the job using:"
echo "  ray job list"
echo "  ray job logs <job-id>"
echo "  Ray dashboard: http://127.0.0.1:8265" 
echo ""
echo "This small test should complete in ~5-10 minutes!"
echo "Look for 'step:1' and 'step:2' completion messages." 