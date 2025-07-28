# 🚀 GRPO Training with Ray - Complete Setup Guide

A distributed training setup for **Group Relative Policy Optimization (GRPO)** using Ray and vLLM, specifically configured for Qwen-2.5-7B model training.

## 📋 **Table of Contents**
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Scripts](#training-scripts)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Directory Structure](#directory-structure)
- [Advanced Usage](#advanced-usage)

---

## 🎯 **Overview**

This repository provides a complete setup for distributed GRPO training using:
- **Ray**: Distributed computing framework
- **vLLM**: High-performance LLM inference engine
- **VERL**: Versatile Efficient Reinforcement Learning library
- **Qwen-2.5-7B**: Target language model

### **Key Features:**
- ✅ **Distributed Training**: 4 GPU support with tensor parallelism
- ✅ **Two Training Modes**: Ray job submission & direct execution
- ✅ **Automatic Environment Setup**: Ray cluster management
- ✅ **Fault Tolerance**: Ray-based recovery and monitoring
- ✅ **Flexible Configuration**: Command-line parameter override

---

## 🔧 **Prerequisites**

### **Hardware Requirements:**
- **GPUs**: 4x NVIDIA L40S (or similar, 48GB+ VRAM recommended)
- **RAM**: 256GB+ system memory
- **Storage**: 100GB+ free space for models and checkpoints

### **Software Requirements:**
- **OS**: Linux (tested on Amazon Linux 2023)
- **Python**: 3.11+ (Python 3.9 has aiohttp compatibility issues)
- **CUDA**: 12.4+
- **Conda/Miniconda**: For environment management

---

## 📦 **Installation**

### **Step 1: Clone Repository**
```bash
cd /home/ec2-user/dataset
git clone <your-repo-url> simpleRL-reason
cd simpleRL-reason
```

### **Step 2: Create Python Environment**
```bash
# Create Python 3.11 environment (CRITICAL: Python 3.9 won't work)
conda create -n mat-verl-py311 python=3.11 -y
conda activate mat-verl-py311
```

### **Step 3: Install Dependencies**
```bash
# Install PyTorch with CUDA 12.4 support
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention
pip3 install flash-attn --no-build-isolation

# Install project dependencies
pip3 install -e .
```

### **Step 4: Setup Directory Structure**
```bash
# Create required directories
mkdir -p /home/ec2-user/dataset/simpleRL-reason/training/models
mkdir -p /home/ec2-user/dataset/simpleRL-reason/training/checkpoints  
mkdir -p /home/ec2-user/dataset/simpleRL-reason/training/logs
mkdir -p /home/ec2-user/dataset/simpleRL-reason/training/simplelr_math_35
```

### **Step 5: Download Model and Prepare Data**
```bash
# Download Qwen-2.5-7B model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen2.5-7B', local_dir='/home/ec2-user/dataset/simpleRL-reason/training/models/Qwen-2.5-7B', local_dir_use_symlinks=False)
"

# Copy your training data
cp /path/to/your/train.parquet /home/ec2-user/dataset/simpleRL-reason/simplelr_math_35/train.parquet
cp /path/to/your/test.parquet /home/ec2-user/dataset/simpleRL-reason/simplelr_math_35/test.parquet
```

---

## 🚀 **Quick Start**

### **Basic Training Command**
```bash
cd /home/ec2-user/dataset/simpleRL-reason
conda activate mat-verl-py311

# Option 1: Ray Job Submit (Recommended)
./start_grpo_ray_job.sh --model_name Qwen-2.5-7B --max_response_length 8192 --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0.0001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.75 --rollout_tp 2 --save_freq 5
```

---

## 📜 **Training Scripts**

### **🔥 Option 1: Ray Job Submit** (`start_grpo_ray_job.sh`)

**Best for:** Production training, job management, monitoring

**Features:**
- ✅ **Job Management**: Submit, monitor, stop jobs via Ray API
- ✅ **Fault Tolerance**: Automatic job recovery
- ✅ **Web Dashboard**: Real-time monitoring at `http://127.0.0.1:8265`
- ✅ **Structured Logging**: Centralized log management
- ✅ **Resource Isolation**: Clean job environment

**Usage:**
```bash
./start_grpo_ray_job.sh [OPTIONS]
```

**Monitoring:**
```bash
# List all jobs
export RAY_ADDRESS="http://127.0.0.1:8265" && ray job list

# View specific job logs
ray job logs <JOB_ID> --follow

# Stop a job
ray job stop <JOB_ID>
ray stop -f

# Open dashboard in browser
# http://127.0.0.1:8265
```

### **⚡ Option 2: Direct Python Execution** (`start_grpo_training.sh`)

**Best for:** Development, debugging, simple setups

**Features:**
- ✅ **Direct Control**: Run training process directly
- ✅ **Simple Debugging**: Console output and process monitoring
- ✅ **Lightweight**: No job submission overhead
- ✅ **Manual Management**: Full control over process lifecycle

**Usage:**
```bash
./start_grpo_training.sh [OPTIONS]
```

**Monitoring:**
```bash
# Check Ray cluster status
export RAY_ADDRESS="172.31.31.243:6379" && ray status

# Monitor training process
ps aux | grep main_ppo

# View GPU usage
nvidia-smi
```

---

## ⚙️ **Configuration**

### **Command Line Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `Qwen-2.5-7B` | Model to train |
| `--max_response_length` | `8192` | Maximum response token length |
| `--train_batch_size` | `1024` | Training batch size |
| `--rollout_n` | `8` | Number of rollout samples per prompt |
| `--kl_loss_coef` | `0.0001` | KL divergence loss coefficient |
| `--entropy_coeffient` | `0.001` | Entropy regularization coefficient |
| `--rollout_gpu_memory_util` | `0.75` | GPU memory utilization for rollout |
| `--rollout_tp` | `2` | Tensor parallel size for rollout |
| `--save_freq` | `5` | Checkpoint save frequency (epochs) |
| `--total_epochs` | `20` | Total training epochs |

### **Example Configurations**

**Small Scale Training:**
```bash
./start_grpo_ray_job.sh --train_batch_size 512 --rollout_n 4 --total_epochs 10
```

**High Quality Training:**
```bash
./start_grpo_ray_job.sh --max_response_length 16384 --rollout_n 16 --kl_loss_coef 0.00005 --total_epochs 50
```

**Memory Optimized:**
```bash
./start_grpo_ray_job.sh --rollout_gpu_memory_util 0.6 --train_batch_size 512 --rollout_tp 4
```

---

## 📊 **Monitoring**

### **Ray Dashboard**
Access the web interface at: `http://127.0.0.1:8265`

**Features:**
- Real-time cluster status
- Job management and logs
- Resource utilization graphs
- Worker node monitoring

### **Command Line Monitoring**

**Check Ray Cluster:**
```bash
export RAY_ADDRESS="172.31.31.243:6379"
ray status
```

**Monitor Training Jobs:**
```bash
export RAY_ADDRESS="http://127.0.0.1:8265"
ray job list
ray job logs <JOB_ID>
```

**System Monitoring:**
```bash
# GPU usage
nvidia-smi

# CPU and memory
htop

# Training process
ps aux | grep main_ppo
```

### **Checkpoints and Logs**

**Checkpoint Location:**
```
/home/ec2-user/dataset/simpleRL-reason/training/checkpoints  verl-grpo_<model>_<config>/
```

**Log Files:**
```
/home/ec2-user/dataset/simpleRL-reason/training/logs
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Ray Cluster Connection Issues**
```bash
# Solution: Restart Ray cluster
ray stop --force
pkill -f ray
rm -rf /tmp/ray*
ray start --head --node-ip-address 172.31.31.243 --num-gpus 4
```

#### **2. aiohttp Compatibility Errors**
```bash
# Solution: Ensure Python 3.11+ is being used
conda activate mat-verl-py311
python --version  # Should show 3.11+
```

#### **3. Stop Token Issues**
The scripts automatically handle Qwen-2.5-7B stop tokens. If you see `stop_token_ids` errors, verify the model path contains "Qwen-2.5" or "qwen-2.5".

#### **4. GPU Memory Issues**
```bash
# Solution: Reduce memory utilization
./start_grpo_ray_job.sh --rollout_gpu_memory_util 0.6 --train_batch_size 512
```

#### **5. Dashboard Not Accessible**
```bash
# Verify dashboard is running on localhost
curl -s http://127.0.0.1:8265/api/jobs
```

### **Debug Mode**
For detailed debugging, run scripts with verbose output:
```bash
export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=debug
./start_grpo_training.sh [OPTIONS]
```

---

## 📁 **Directory Structure**

```
simpleRL-reason/
├── README.md                          # This file
├── start_grpo_ray_job.sh              # Ray job submit script
├── start_grpo_training.sh             # Direct Python execution script  
├── train_grpo_math_tune_ray.sh        # Original training script (updated)
├── requirements.txt                   # Python dependencies
├── verl/                              # VERL library source
├── examples/                          # Example configurations
└── simplelr_math_35/                  # Training data
    ├── train.parquet                  # Training dataset
    └── test.parquet                   # Validation dataset

---

## 🔬 **Advanced Usage**

### **Custom Model Training**
```bash
# Train with your own model
./start_grpo_ray_job.sh --model_name YourModel-7B --max_response_length 4096
```

### **Multi-Node Training** (Future Enhancement)
```bash
# Start additional worker nodes
ray start --address='HEAD_NODE_IP:6379' --num-gpus 4
```

### **Custom Data Format**
Ensure your parquet files have the required columns:
- `prompt`: Input prompt text
- `chosen`: Preferred response (for GRPO training)

### **Environment Variables**
```bash
export WANDB_API_KEY="your_key"        # Enable Weights & Biases logging
export NCCL_DEBUG=WARN                 # NCCL debugging
export CUDA_VISIBLE_DEVICES=0,1,2,3    # GPU selection
```

---

## 🤝 **Support**

**For issues:**
1. Check [Troubleshooting](#troubleshooting) section
2. Verify all prerequisites are met
3. Check Ray dashboard for cluster status
4. Review training logs for specific errors

**System Requirements Verification:**
```bash
# Check CUDA
nvidia-smi

# Check Python environment
conda activate mat-verl-py311 && python --version

# Check Ray installation
ray --version

# Test Ray cluster
ray start --head --num-gpus 4 && ray status && ray stop
```

---

## 📄 **License**

[Add your license information here]

---

**🎉 Happy Training!** 🚀


