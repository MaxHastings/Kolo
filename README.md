# Kolo

Tired of spending hours setting up your LLM fine-tuning environment? Kolo automates the entire process, getting you up and running in just 5 minutes with zero hassle. Get started instantly—whether you're an AI researcher, developer, or just experimenting with fine-tuning, Kolo makes it effortless.

## 🛠 Tools Installed

Kolo is built using a powerful stack of LLM tools:

- [Unsloth](https://github.com/unslothai/unsloth) – Open-source LLM fine-tuning; faster training, lower VRAM.
- [Torchtune](https://github.com/pytorch/torchtune) – Native PyTorch library LLM fine-tuning which supports AMD GPU and CPU fine tuning.
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) – C/C++ converting and quantization of LLMs into GGUFs for easy testing and deployment.
- [Ollama](https://ollama.ai/) – Portable, user-friendly LLM model management and deployment software.
- [Docker](https://www.docker.com/) – Containerized environment to automatically setup the entire LLM development environment with the necessary tools and dependencies automatically installed along with scripts to make fine tuning and testing easy.
- [Open WebUI](https://github.com/open-webui/open-webui) – Self-hosted web interface for LLM testing.

## Recommended System Requirements

- Operating System: Windows 10 or later, or Linux
- Graphics Card: Nvidia GPU with CUDA 12.1 support and at least 8GB of VRAM
- AMD GPU Users: Linux is required; Windows WSL2 does not support ROCM.
- Memory: 16GB or more of system RAM

May work on other systems, your results may vary. Let us know!

## Issues or Feedback

Join our [Discord group](https://discord.gg/Ewe4hf5x3n)!

## 🏃 Getting Started

### 1️⃣ Install Dependencies

#### 🖥️ Windows Requirements

Ensure [HyperV](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v?pivots=windows) is installed.

Ensure [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) is installed; alternative [guide](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers).

Ensure [Docker Desktop](https://docs.docker.com/get-docker/) is installed.

#### 🐧 Linux Requirements

Ensure [Docker Desktop](https://docs.docker.com/get-docker/) is installed. Or [Docker CLI](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

#### AMD Requirements

Install [ROCM](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) on Linux.

### Clone the repository 

```bash
git clone git@github.com:MaxHastings/Kolo.git
cd Kolo
```
### Make all bash scripts executable

```bash
chmod +x *.sh
```

### 2️⃣ Build the Image

To build the image, run:

```bash
# For NVIDIA GPU
./build_image.sh --type cuda

# For AMD GPU
./build_image.sh --type rocm

# For CPU only
./build_image.sh --type cpu
# or simply
./build_image.sh
```

### 3️⃣ Run the Container

```bash
./create_and_run_container.sh
```

For subsequent runs:
```bash
./run_container.sh
```

### 4️⃣ Copy Training Data

```bash
./copy_training_data.sh -f examples/God.jsonl -d data.jsonl
```

Don't have training data? Check out our synthetic QA [data generation guide](GenerateTrainingDataGuide.md)!

### 5️⃣ Train Model

#### Using Unsloth

```bash
./train_model_unsloth.sh --OutputDir "GodOutput" --Quantization "Q4_K_M" --TrainData "data.jsonl"
```

All available parameters

```bash
./train_model_unsloth.sh --Epochs 3 --LearningRate 1e-4 --TrainData "data.jsonl" -
-BaseModel "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" --ChatTemplate "llama-3.1" --LoraRank 16 --LoraAlpha 16 --LoraDropou
t 0 --MaxSeqLength 1024 --WarmupSteps 10 --SaveSteps 500 --SaveTotalLimit 5 --Seed 1337 --SchedulerType "linear" --Batch
Size 2 --OutputDir "GodOutput" --Quantization "Q4_K_M" --WeightDecay 0
```

#### Using Torchtune

Requirements: Create a [Hugging Face](https://huggingface.co/) account and create a token. You will also need to get permission from Meta to use their models. Search the Base Model name on Hugging Face website and get access before training.

If you are using an AMD GPU, use the following command instead:

```bash
./train_model_torchtune_gpu.sh --gpuArch gfx90a --outputDir GodOutput --quantization Q4_K_M --trainData data.json --hfToken "YOUR_HF_TOKEN"
```

All available parameters

```bash
./train_model_torchtune_gpu.sh --gpuarch gfx90a --hfToken "your_token" --epochs 3 --learningRate 1e-4 --trainData "data.json" --baseModel "Meta-llama/Llama-3.2-1B-Instruct" --loraRank 16 --loraAlpha 16 --loraDropout 0 --maxSeqLength 1024 --warmupSteps 10 --seed 1337 --schedulerType "cosine" --batchSize 2 --outputDir "GodOutput" --quantization "Q4_K_M" --weightDecay 0
```

For CPU:

```bash
./train_model_torchtune.sh --OutputDir "GodOutput" --Quantization "Q4_K_M" --TrainData "data.json" -HfToken "your_token"
```

All available parameters

```bash
 ./train_model_torchtune.sh --HfToken "your_token" --Epochs 3 --LearningRate 1e-4 --TrainData "data.json" --BaseModel "Meta-llama/Llama-3.2-1B-Instruct" --LoraRank 16 --LoraAlpha 16 --LoraDropout 0 --MaxSeqLength 1024 --WarmupSteps 10 --Seed 1337 --SchedulerType "cosine" --BatchSize 2 --OutputDir "GodOutput" --Quantization "Q4_K_M" --WeightDecay 0
```


Note: If re-training with the same OutputDir, delete the existing directory first:

```bash
./delete_model.sh "GodOutput" "unsloth|torchtune"
```

For more information about fine tuning parameters please refer to the [Fine Tune Training Guide](https://github.com/MaxHastings/Kolo/blob/main/FineTuningGuide.md).

### 6️⃣ Install Model

#### Using Unsloth

```bash
./install_model.sh "God" --tool "unsloth" --output-dir "GodOutput" --quantization "Q4_K_M
```

#### Using Torchtune

```bash
./install_model.sh "God" --tool "torchtune" --output-dir "GodOutput" --quantization "Q4_K_M
```

### 7️⃣ Test Model

Open your browser and navigate to [localhost:8080](http://localhost:8080/)

![Open WebUI Demo](https://github.com/open-webui/open-webui/raw/main/demo.gif)

### Other Commands

Uninstalls the Model from Ollama.

```bash
./uninstall_model.sh "God"
```

Lists all models installed on Ollama and the training model directories for both torchtune and unsloth.

```bash
./list_models.sh
```

## 🔧 Advanced Users

### SSH Access

To quickly SSH into the Kolo container for installing additional tools or running scripts directly:

```bash
./connect.sh
```

If prompted for a password, use:

```bash
password 123
```

Alternatively, you can connect manually via SSH:

```bash
ssh root@localhost -p 2222
```

### WinSCP (SFTP Access)

You can use [WinSCP](https://winscp.net/eng/index.php) or any other SFTP file manager to access the Kolo container’s file system. This allows you to manage, modify, add, or remove scripts and files easily.

Connection Details:

- Host: localhost
- Port: 2222
- Username: root
- Password: 123

This setup ensures you can easily transfer files between your local machine and the container.
