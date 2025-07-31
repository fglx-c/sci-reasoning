#!/usr/bin/env python3

import sys
import os

print("=== GRPO Training Environment Setup Verification ===\n")

# Test basic imports
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import verl
    print("✓ VERL imported successfully")
except Exception as e:
    print(f"✗ VERL import failed: {e}")
    print("  Make sure you're in the sci-res conda environment and VERL is installed")
    sys.exit(1)

try:
    import ray
    print(f"✓ Ray version: {ray.__version__}")
except Exception as e:
    print(f"✗ Ray import failed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers import failed: {e}")
    sys.exit(1)

try:
    import networkx as nx
    print(f"✓ NetworkX version: {nx.__version__}")
except Exception as e:
    print(f"✗ NetworkX import failed: {e}")
    sys.exit(1)

try:
    import flash_attn
    print("✓ Flash Attention imported successfully")
except Exception as e:
    print(f"⚠ Flash Attention import failed: {e}")
    print("  This is optional but recommended for performance")

# Test data files existence
train_file = "/scratch/miaosenc/sci-reasoning/training/rlvr-training/simplelr_math_35_small/train.parquet"
val_file = "/scratch/miaosenc/sci-reasoning/training/rlvr-training/simplelr_math_35_small/test.parquet"

if os.path.exists(train_file):
    print(f"✓ Training file exists: {train_file}")
else:
    print(f"⚠ Training file missing: {train_file}")

if os.path.exists(val_file):
    print(f"✓ Validation file exists: {val_file}")
else:
    print(f"⚠ Validation file missing: {val_file}")

# Test model paths
model_path = "/scratch/miaosenc/sci-reasoning/training/models/OpenReasoning-Nemotron-1.5B"
reward_model_path = "/scratch/miaosenc/sci-reasoning/training/models/general-verifier"

if os.path.exists(model_path):
    print(f"✓ Model path exists: {model_path}")
else:
    print(f"⚠ Model path missing: {model_path}")

if os.path.exists(reward_model_path):
    print(f"✓ Reward model path exists: {reward_model_path}")
else:
    print(f"⚠ Reward model path missing: {reward_model_path}")

# Check environment variables
print(f"\n=== Environment Variables ===")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"NCCL_NET_OFI_DISABLE: {os.environ.get('NCCL_NET_OFI_DISABLE', 'Not set')}")
print(f"no_proxy: {os.environ.get('no_proxy', 'Not set')}")

print(f"\n🎉 Setup verification completed!")
print(f"Environment appears ready for GRPO training.")

# Instructions for next steps
print(f"\nNext steps:")
print(f"1. Run: source setup_env.sh")
print(f"2. Start training with the provided commands in HollySetUp.md")
