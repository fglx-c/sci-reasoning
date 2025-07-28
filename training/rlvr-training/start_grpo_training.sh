#!/bin/bash

# GRPO Training Script for Qwen-2.5-7B
# This script starts Ray cluster and runs GRPO training directly

set -e  # Exit on any error

USER_ENV=`whoami`
echo "Starting GRPO training for user: $USER_ENV"

# Environment setup
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Default parameters (can be overridden by command line arguments)
MODEL_NAME="OpenReasoning-Nemotron-1.5B"
MAX_RESPONSE_LENGTH=8192
TRAIN_BATCH_SIZE=1024
ROLLOUT_N=8
KL_LOSS_COEF=0.0001
ENTROPY_COEF=0.001
ROLLOUT_GPU_MEMORY_UTIL=0.75
ROLLOUT_TP=2
SAVE_FREQ=5
TOTAL_EPOCHS=20

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
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Paths
MODEL_PATH="/home/ec2-user/dataset/training/models/$MODEL_NAME"
TRAIN_FILE="/home/ec2-user/dataset/simpleRL-reason/simplelr_math_35/train.parquet"
VAL_FILE="/home/ec2-user/dataset/simpleRL-reason/simplelr_math_35/test.parquet"
CHECKPOINT_DIR="/home/ec2-user/dataset/training/checkpoints/verl-grpo_${MODEL_NAME}_max_response${MAX_RESPONSE_LENGTH}_batch${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_klcoef${KL_LOSS_COEF}_entcoef${ENTROPY_COEF}_simplelr_math_35"

echo "==========================================="
echo "GRPO Training Configuration:"
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
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "==========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "mat-verl-py311" ]]; then
    echo "Activating mat-verl-py311 environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate mat-verl-py311
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please make sure the model is downloaded."
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

# Check Ray status
echo "Checking Ray cluster status..."
if ! ray status > /dev/null 2>&1; then
    echo "Ray cluster not running. Starting Ray..."
    ray start --head --node-ip-address 127.0.0.1 --num-gpus 4
    sleep 5
else
    echo "Ray cluster is already running."
fi

# Verify Ray is working
if ! ray status > /dev/null 2>&1; then
    echo "Error: Failed to start Ray cluster"
    exit 1
fi

echo "Starting GRPO training..."
echo "Log will be saved to: ${CHECKPOINT_DIR}/training.log"

# Calculate max tokens
max_num_batched_tokens=$((1024 + $MAX_RESPONSE_LENGTH + 1000))

# Start training with all parameters
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.micro_rollout_batch_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=verl_train \
    trainer.remove_previous_ckpt=False \
    trainer.experiment_name="verl-grpo_${MODEL_NAME}_max_response${MAX_RESPONSE_LENGTH}_batch${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_klcoef${KL_LOSS_COEF}_entcoef${ENTROPY_COEF}_simplelr_math_35" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.remove_clip=False \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=5 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    2>&1 | tee "${CHECKPOINT_DIR}/training.log"

echo "Training completed or stopped." 