#!/bin/bash
HARDWARE_TYPE=$(cat /tmp/hardware_type.txt)
echo "Detected hardware: $HARDWARE_TYPE"

if [ "$HARDWARE_TYPE" = "nvidia" ]; then
    echo "Setting up NVIDIA GPU environment"
    # Install CUDA dependencies
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-1 && rm -rf /var/lib/apt/lists/*
    # NVIDIA-specific setup
    /opt/conda/bin/conda install -y pytorch-cuda=12.1 cudatoolkit=11.7.0 -c pytorch -c nvidia
    /opt/conda/bin/conda run -n kolo_env pip install xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
elif [ "$HARDWARE_TYPE" = "amd" ]; then
    echo "Setting up AMD GPU environment"
    # AMD-specific setup
    apt-get update && apt-get install -y wget gnupg2 && \
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4.2/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && apt-get install -y rocm-libs hip-runtime-amd && apt-get clean
    /opt/conda/bin/conda run -n kolo_env pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4
    /opt/conda/bin/conda run -n kolo_env pip install --pre torchaudio --index-url https://download.pytorch.org/whl/rocm5.4 || echo "torchaudio not available for ROCm 5.4"
else
    echo "Setting up CPU environment"
    # CPU-specific setup (default)
fi