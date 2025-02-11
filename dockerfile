# Use Ubuntu as the base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y openssh-server sudo build-essential curl git wget vim && \
    rm -rf /var/lib/apt/lists/*

# Add NodeSource repository for Node.js v18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -

# Install Node.js (v18.x) and npm from NodeSource
RUN apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Create the SSH daemon run directory.
RUN mkdir /var/run/sshd

# Set the root password and update SSH config to permit root login.
RUN echo 'root:123' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN mkdir -p /workspace

# Install Anaconda3:
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O anaconda.sh && \
    bash anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh

# Create Kolo env
RUN /opt/conda/bin/conda create -y --name kolo_env python=3.10

# Run Kolo env
SHELL ["/opt/conda/bin/conda", "run", "-n", "kolo_env", "/bin/bash", "-c"]

RUN conda config --set remote_read_timeout_secs 86400

# Install torchtune
RUN pip install torch torchvision torchao
RUN pip install torchtune

# Create a Conda environment and install PyTorch with CUDA support and xformers
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers && conda clean -afy

# Install unsloth and additional ML/utility packages.
RUN pip config set global.timeout 86400
RUN pip install numpy datasets
RUN pip install unsloth vllm
RUN pip install --upgrade pillow
RUN pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
RUN pip install --no-deps trl peft accelerate bitsandbytes
RUN pip install transformers

# Upgrade Xformers
RUN pip install xformers --upgrade

# Create Open-webui env
RUN /opt/conda/bin/conda create -y --name openwebui_env python=3.11

# Run openwebui env
SHELL ["/opt/conda/bin/conda", "run", "-n", "openwebui_env", "/bin/bash", "-c"]

#Install Open-webui
RUN pip install git+https://github.com/open-webui/open-webui.git@b72150c881955721a63ae7f4ea1b9ea293816fc1

SHELL ["/bin/bash", "-c"]

# Install Ollama.
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set the working directory (optional).
WORKDIR /app

# Create a volume for persistent data.
VOLUME /var/kolo_data

RUN apt-get update && \
    apt-get install -y openssh-server supervisor && \
    rm -rf /var/lib/apt/lists/*

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Init the Conda env
RUN /opt/conda/bin/conda init bash

# Update ~/.bashrc
RUN echo '# activate conda env' | tee -a ~/.bashrc
RUN echo 'conda activate kolo_env' | tee -a ~/.bashrc
RUN echo '' | tee -a ~/.bashrc

# Expose necessary ports
EXPOSE 22 8080

RUN apt-get update && apt-get install -y cmake && apt-get clean

RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    cmake -B build && \
    cmake --build build --config Release

RUN mv llama.cpp/build/bin/llama-quantize llama.cpp/

# Copy scripts
COPY scripts /app/

# Copy torchtune configs
COPY torchtune /app/torchtune

# Set the entrypoint to start supervisord
CMD ["/usr/bin/supervisord"]