#!/usr/bin/env python3

import sys
import os

# Test basic imports
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import verl
    print("✓ VERL imported successfully")
except Exception as e:
    print(f"✗ VERL import failed: {e}")
    sys.exit(1)

# Test data files
train_file = "/scratch/miaosenc/sci-reasoning/training/rlvr-training/simplelr_math_35_small/train.parquet"
val_file = "/scratch/miaosenc/sci-reasoning/training/rlvr-training/simplelr_math_35_small/test.parquet"

if os.path.exists(train_file):
    print(f"✓ Training file exists: {train_file}")
else:
    print(f"✗ Training file missing: {train_file}")
    sys.exit(1)

if os.path.exists(val_file):
    print(f"✓ Validation file exists: {val_file}")
else:
    print(f"✗ Validation file missing: {val_file}")
    sys.exit(1)

# Test model path
model_path = "/scratch/miaosenc/sci-reasoning/training/models/OpenReasoning-Nemotron-1.5B"
if os.path.exists(model_path):
    print(f"✓ Model path exists: {model_path}")
else:
    print(f"✗ Model path missing: {model_path}")
    sys.exit(1)

print("\n🎉 All checks passed! Ready for training.") 