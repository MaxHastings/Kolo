#!/bin/bash

# Script to set up hardware-specific dependencies
# Usage: ./setup_hardware.sh <hardware_type>

HARDWARE_TYPE="${1:-cpu}"
echo "Setting up environment for hardware type: $HARDWARE_TYPE"

# Activate conda environment
source /opt/conda/bin/activate kolo_env

case "$HARDWARE_TYPE" in
  "cpu")
    echo "Installing CPU-specific packages..."
    # CPU-specific PyTorch installation
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    pip install torchao==0.8.0 torchtune==0.5.0
    ;;
    
  "cuda")
    echo "Installing NVIDIA CUDA-specific packages..."
    # Install CUDA dependencies
    apt-get update && apt-get install -y --no-install-recommends \
        nvidia-cuda-toolkit \
        nvidia-cuda-toolkit-gcc && \
        rm -rf /var/lib/apt/lists/*
        
    # CUDA-specific PyTorch installation
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
    pip install torchao==0.8.0 torchtune==0.5.0
    
    # Install NVIDIA-specific optimizations
    pip install triton flash-attn
    ;;
    
  "rocm")
    echo "Installing AMD ROCm-specific packages..."
    # Install ROCm dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        gnupg \
        python3-pip && \
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        rocm-dev \
        hipblas \
        miopen-hip \
        rocm-libs && \
    rm -rf /var/lib/apt/lists/*
    
    # ROCm-specific PyTorch installation
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm5.6
    pip install torchao==0.8.0 torchtune==0.5.0
    ;;
    
  *)
    echo "Unknown hardware type: $HARDWARE_TYPE"
    echo "Defaulting to CPU setup..."
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    pip install torchao==0.8.0 torchtune==0.5.0
    ;;
esac

echo "Hardware-specific setup complete for $HARDWARE_TYPE"