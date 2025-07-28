#!/usr/bin/env python3
"""
Create a small test dataset (1/10000 size) for quick training validation.
"""

import pandas as pd
import os
import shutil

def create_small_dataset():
    # Paths
    original_train = "simplelr_math_35/train.parquet"
    original_test = "simplelr_math_35/test.parquet"
    
    small_dir = "simplelr_math_35_small"
    small_train = f"{small_dir}/train.parquet"
    small_test = f"{small_dir}/test.parquet"
    
    # Load original datasets
    print("Loading original datasets...")
    train_df = pd.read_parquet(original_train)
    test_df = pd.read_parquet(original_test)
    
    print(f"Original train size: {len(train_df)}")
    print(f"Original test size: {len(test_df)}")
    
    # Calculate small sizes (1/10000)
    train_small_size = max(1, len(train_df) // 10000)  # At least 1 sample
    test_small_size = max(1, len(test_df) // 10000)   # At least 1 sample
    
    print(f"Small train size: {train_small_size}")
    print(f"Small test size: {test_small_size}")
    
    # Sample randomly
    train_small = train_df.sample(n=train_small_size, random_state=42)
    test_small = test_df.sample(n=test_small_size, random_state=42)
    
    # Create output directory
    if os.path.exists(small_dir):
        shutil.rmtree(small_dir)
    os.makedirs(small_dir)
    
    # Save small datasets
    train_small.to_parquet(small_train)
    test_small.to_parquet(small_test)
    
    print(f"\nSmall dataset created in '{small_dir}/':")
    print(f"  - train.parquet: {len(train_small)} samples")
    print(f"  - test.parquet: {len(test_small)} samples")
    
    # Show sample data
    print(f"\nSample from small train dataset:")
    print(f"Question: {train_small.iloc[0]['question'][:100]}...")
    print(f"Answer: {train_small.iloc[0]['answer']}")

if __name__ == "__main__":
    create_small_dataset() 