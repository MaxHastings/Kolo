# Unified Dockerfile for CPU/NVIDIA/AMD environments
FROM ubuntu:22.04

# Set the DEBIAN frontend to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install base packages
RUN apt-get update && \
    apt-get install -y openssh-server sudo build-essential curl git wget vim cmake supervisor \
    lshw pciutils python3-minimal python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js v18.20.6
RUN curl -fsSL https://deb.nodesource.com/node_18.x/pool/main/n/nodejs/nodejs_18.20.6-1nodesource1_amd64.deb -o nodejs.deb && \
    dpkg -i nodejs.deb && \
    rm -f nodejs.deb

# Create the SSH daemon run directory
RUN mkdir /var/run/sshd

# Set the root password and update SSH config
RUN echo 'root:123' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Create workspace directory
RUN mkdir -p /workspace

# Install Anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O anaconda.sh && \
    bash anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh

# Create Kolo env
RUN /opt/conda/bin/conda create -y --name kolo_env python=3.10

# Set Conda timeout
RUN /opt/conda/bin/conda config --set remote_read_timeout_secs 86400

# Copy hardware detection script and setup scripts
COPY scripts/detect_hardware.py /tmp/
COPY scripts/setup_env.sh /tmp/
COPY scripts/run_detection.sh /app/

# Make scripts executable
RUN chmod +x /tmp/detect_hardware.py /tmp/setup_env.sh /app/run_detection.sh

# Run hardware detection and store the result
RUN python3 /tmp/detect_hardware.py || echo "cpu" > /tmp/hardware_type.txt

# Run environment setup script
RUN /tmp/setup_env.sh

# Install common packages in kolo_env
SHELL ["/opt/conda/bin/conda", "run", "-n", "kolo_env", "/bin/bash", "-c"]

# Install PyTorch and related packages
RUN pip install torch==2.6.0 torchvision==0.21.0 torchao==0.8.0 torchtune==0.5.0

# Set a long timeout for pip commands
RUN pip config set global.timeout 86400

# Install packages with exact version pins
RUN pip install numpy==2.2.3 datasets==3.3.2

# Install unsloth from a specific commit
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@038e6d4c8d40207a87297ab3aaf787c19b1006d1"

# Install additional ML/utility packages with version pins
RUN pip install --no-deps trl==0.14.0 peft==0.14.0 accelerate==1.4.0 bitsandbytes==0.45.3

# Freeze transformers version
RUN pip install transformers==4.49.0

# Install OpenAI with a fixed version
RUN pip install openai==1.64.0

# Create Open-webui env
RUN /opt/conda/bin/conda create -y --name openwebui_env python=3.11

# Run openwebui env
SHELL ["/opt/conda/bin/conda", "run", "-n", "openwebui_env", "/bin/bash", "-c"]

# Install Open-webui
RUN NODE_OPTIONS="--max-old-space-size=8096" pip install git+https://github.com/open-webui/open-webui.git@b72150c881955721a63ae7f4ea1b9ea293816fc1

# Switch back to Bash shell
SHELL ["/bin/bash", "-c"]

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.12 sh

# Set the working directory
WORKDIR /app

# Create a volume for persistent data
VOLUME /var/kolo_data

# Init the Conda env
RUN /opt/conda/bin/conda init bash

# Update ~/.bashrc to auto-activate the kolo_env
RUN echo '# activate conda env' >> ~/.bashrc && \
    echo 'conda activate kolo_env' >> ~/.bashrc && \
    echo '' >> ~/.bashrc

# Expose necessary ports
EXPOSE 22 8080

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    git checkout a82c9e7c23ef6db48cebfa194dc9cebbc4ac3552 && \
    cmake -B build && \
    cmake --build build --config Release

RUN mv llama.cpp/build/bin/llama-quantize llama.cpp/

# Copy remaining scripts and torchtune configurations
COPY scripts /app/scripts/
COPY torchtune /app/torchtune/

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Use the detection script as entrypoint
ENTRYPOINT ["/app/run_detection.sh"]

# Set the command to start supervisord
CMD ["/usr/bin/supervisord"]