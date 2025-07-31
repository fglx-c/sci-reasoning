#!/bin/bash

# Get machine IP address dynamically
MACHINE_IP=$(hostname -I | awk '{print $1}')

# Environment setup for Holly system
export PYTHONUNBUFFERED=1
export RAY_ADDRESS="${MACHINE_IP}:6379"
export RAY_JOB_ADDRESS="http://127.0.0.1:8265"

# Bypass proxy for localhost (Holly system specific)
export no_proxy="localhost,127.0.0.1"

# Use conda's libstdc++ to fix GLIBCXX version issues
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Prevent NCCL from hanging on systems without EFA/OFI
export NCCL_NET_OFI_DISABLE=1

echo "Environment configured for machine IP: $MACHINE_IP"
echo "Ray Address: $RAY_ADDRESS"
echo "Ray Job Address: $RAY_JOB_ADDRESS"
echo "Proxy bypass: $no_proxy"
echo "LD_LIBRARY_PATH configured for conda environment"
