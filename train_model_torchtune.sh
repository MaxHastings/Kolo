#!/bin/bash

# Default values
CONTAINER_NAME="kolo_container"
BASE_MODEL="Meta-llama/Llama-3.2-1B-Instruct"
QUANTIZATION="Q4_K_M"
OUTPUT_DIR="outputs"
USE_CHECKPOINT=false
FAST_TRANSFER=false
GPU_ARCH=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --Epochs) EPOCHS="$2"; shift ;;
        --LearningRate) LEARNING_RATE="$2"; shift ;;
        --TrainData) TRAIN_DATA="$2"; shift ;;
        --BaseModel) BASE_MODEL="$2"; shift ;;
        --LoraRank) LORA_RANK="$2"; shift ;;
        --LoraAlpha) LORA_ALPHA="$2"; shift ;;
        --LoraDropout) LORA_DROPOUT="$2"; shift ;;
        --MaxSeqLength) MAX_SEQ_LENGTH="$2"; shift ;;
        --WarmupSteps) WARMUP_STEPS="$2"; shift ;;
        --Seed) SEED="$2"; shift ;;
        --SchedulerType) SCHEDULER_TYPE="$2"; shift ;;
        --BatchSize) BATCH_SIZE="$2"; shift ;;
        --OutputDir) OUTPUT_DIR="$2"; shift ;;
        --Quantization) QUANTIZATION="$2"; shift ;;
        --WeightDecay) WEIGHT_DECAY="$2"; shift ;;
        --UseCheckpoint) USE_CHECKPOINT=true ;;
        --HfToken) HF_TOKEN="$2"; shift ;;
        --FastTransfer) FAST_TRANSFER=true ;;
        --GpuArch) GPU_ARCH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Log received parameters
echo -e "\e[36mParameters passed:\e[0m"
[[ -n "$EPOCHS" ]] && echo "Epochs: $EPOCHS"
[[ -n "$LEARNING_RATE" ]] && echo "LearningRate: $LEARNING_RATE"
[[ -n "$TRAIN_DATA" ]] && echo "TrainData: $TRAIN_DATA"
[[ -n "$BASE_MODEL" ]] && echo "BaseModel: $BASE_MODEL"
[[ -n "$LORA_RANK" ]] && echo "LoraRank: $LORA_RANK"
[[ -n "$LORA_ALPHA" ]] && echo "LoraAlpha: $LORA_ALPHA"
[[ -n "$LORA_DROPOUT" ]] && echo "LoraDropout: $LORA_DROPOUT"
[[ -n "$MAX_SEQ_LENGTH" ]] && echo "MaxSeqLength: $MAX_SEQ_LENGTH"
[[ -n "$WARMUP_STEPS" ]] && echo "WarmupSteps: $WARMUP_STEPS"
[[ -n "$SEED" ]] && echo "Seed: $SEED"
[[ -n "$SCHEDULER_TYPE" ]] && echo "SchedulerType: $SCHEDULER_TYPE"
[[ -n "$BATCH_SIZE" ]] && echo "BatchSize: $BATCH_SIZE"
[[ -n "$OUTPUT_DIR" ]] && echo "OutputDir: $OUTPUT_DIR"
[[ -n "$QUANTIZATION" ]] && echo "Quantization: $QUANTIZATION"
[[ -n "$WEIGHT_DECAY" ]] && echo "WeightDecay: $WEIGHT_DECAY"
if $USE_CHECKPOINT; then
    echo "UseCheckpoint: Enabled"
else
    echo "UseCheckpoint: Disabled"
fi

# Log GPU mode or fast transfer based on parameters
if [[ -n "$GPU_ARCH" ]]; then
    echo -e "\e[36mGPU Architecture: $GPU_ARCH\e[0m"
else
    if $FAST_TRANSFER; then
        echo -e "\e[36mFastTransfer: Enabled (HF_HUB_ENABLE_HF_TRANSFER=1)\e[0m"
    else
        echo -e "\e[36mFastTransfer: Disabled (HF_HUB_ENABLE_HF_TRANSFER=0)\e[0m"
    fi
fi

# Define the Docker container name and check if it is running
container_running=$(docker ps --format "{{.Names}}" | grep -w "$CONTAINER_NAME")
if [[ -z "$container_running" ]]; then
    echo -e "\e[31mError: Container '$CONTAINER_NAME' is not running.\e[0m"
    exit 1
fi

# --- Define BaseModel to config mapping ---
declare -A config_map
config_map["Meta-llama/Llama-3.1-8B-Instruct"]="/app/torchtune/configs/llama3_1/8B_qlora_single_device.yaml"
config_map["Meta-llama/Llama-3.2-3B-Instruct"]="/app/torchtune/configs/llama3_2/3B_qlora_single_device.yaml"
config_map["Meta-llama/Llama-3.2-1B-Instruct"]="/app/torchtune/configs/llama3_2/1B_qlora_single_device.yaml"

# Retrieve the configuration value based on the provided BaseModel
if [[ -n "${config_map[$BASE_MODEL]}" ]]; then
    config_value="${config_map[$BASE_MODEL]}"
else
    echo -e "\e[31mError: The specified BaseModel '$BASE_MODEL' was not found in the configuration mapping.\e[0m"
    exit 1
fi

echo -e "\e[36mUsing configuration: $config_value for BaseModel: $BASE_MODEL\e[0m"

# --- Begin BaseModel download step ---
if [[ -z "$HF_TOKEN" ]]; then
    echo -e "\e[31mError: Hugging Face token must be provided.\e[0m"
    exit 1
fi

hf_transfer_value=0
if $FAST_TRANSFER; then
    hf_transfer_value=1
fi

download_command="export HF_HUB_ENABLE_HF_TRANSFER=$hf_transfer_value && source /opt/conda/bin/activate kolo_env && tune download '$BASE_MODEL' --ignore-patterns 'original/consolidated.00.pth' --hf-token '$HF_TOKEN'"

echo -e "\e[33mDownloading BaseModel using command:\e[0m"
echo -e "\e[33m$download_command\e[0m"

if ! docker exec -it $CONTAINER_NAME /bin/bash -c "$download_command"; then
    echo -e "\e[31mFailed to download BaseModel.\e[0m"
    exit 1
else
    echo -e "\e[32mBaseModel downloaded successfully!\e[0m"
fi

# --- Begin torchtune run ---
# Build the base torchtune command string using the configuration from the mapping
if [[ -n "$GPU_ARCH" ]]; then
    # AMD GPU branch: set HIP alloc conf and ROCm arch
    command="export PYTORCH_HIP_ALLOC_CONF='garbage_collection_threshold:0.8,max_split_size_mb:512' && PYTORCH_ROCM_ARCH=$GPU_ARCH source /opt/conda/bin/activate kolo_env && tune run lora_finetune_single_device --config $config_value"
else
    # Default branch
    command="source /opt/conda/bin/activate kolo_env && tune run lora_finetune_single_device --config $config_value"
fi

# Append dynamic parameters with defaults
if [[ -n "$EPOCHS" ]]; then
    command+=" epochs=$EPOCHS"
else
    command+=" epochs=3"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    command+=" batch_size=$BATCH_SIZE"
else
    command+=" batch_size=1"
fi

if [[ -n "$TRAIN_DATA" ]]; then
    command+=" dataset.data_files='$TRAIN_DATA'"
else
    command+=" dataset.data_files=./data.json"
fi

# Fixed dataset parameters
command+=" dataset._component_=torchtune.datasets.chat_dataset"
command+=" dataset.source=json"
command+=" dataset.conversation_column=conversations"
command+=" dataset.conversation_style=sharegpt"

if [[ -n "$LORA_RANK" ]]; then
    command+=" model.lora_rank=$LORA_RANK"
else
    command+=" model.lora_rank=16"
fi

if [[ -n "$LORA_ALPHA" ]]; then
    command+=" model.lora_alpha=$LORA_ALPHA"
else
    command+=" model.lora_alpha=16"
fi

if [[ -n "$LORA_DROPOUT" ]]; then
    command+=" model.lora_dropout=$LORA_DROPOUT"
fi

if [[ -n "$LEARNING_RATE" ]]; then
    command+=" optimizer.lr=$LEARNING_RATE"
else
    command+=" optimizer.lr=1e-4"
fi

if [[ -n "$MAX_SEQ_LENGTH" ]]; then
    command+=" tokenizer.max_seq_len=$MAX_SEQ_LENGTH"
fi

if [[ -n "$WARMUP_STEPS" ]]; then
    command+=" lr_scheduler.num_warmup_steps=$WARMUP_STEPS"
else
    command+=" lr_scheduler.num_warmup_steps=100"
fi

if [[ -n "$SEED" ]]; then
    command+=" seed=$SEED"
fi

if [[ -n "$SCHEDULER_TYPE" ]]; then
    command+=" lr_scheduler._component_=torchtune.training.lr_schedulers.get_${SCHEDULER_TYPE}_schedule_with_warmup"
else
    command+=" lr_scheduler._component_=torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup"
fi

if [[ -n "$WEIGHT_DECAY" ]]; then
    command+=" optimizer.weight_decay=$WEIGHT_DECAY"
else
    command+=" optimizer.weight_decay=0.01"
fi

if $USE_CHECKPOINT; then
    command+=" resume_from_checkpoint=True"
else
    command+=" resume_from_checkpoint=False"
fi
command+=" dtype=fp32"
command+=" device=cpu"
command+=" enable_activation_offloading=False"
command+=" compile=False"
# Set the output directory; default is "outputs"
FULL_OUTPUT_DIR="/var/kolo_data/torchtune/$OUTPUT_DIR"
command+=" output_dir='$FULL_OUTPUT_DIR'"

# Log a note on quantization if provided
if [[ -n "$QUANTIZATION" ]]; then
    echo "Note: Quantization parameter '$QUANTIZATION' is provided and will be used for quantization."
fi

echo -e "\e[33mExecuting torchtune command inside container '$CONTAINER_NAME':\e[0m"
echo -e "\e[33m$command\e[0m"

if ! docker exec -it $CONTAINER_NAME /bin/bash -c "$command"; then
    echo -e "\e[31mFailed to execute torchtune run.\e[0m"
    exit 1
else
    echo -e "\e[32mTorchtune run completed successfully!\e[0m"
fi

# --- Begin post-run merging steps ---
find_epoch_cmd="ls -d ${FULL_OUTPUT_DIR}/epoch_* 2>/dev/null | sort -V | tail -n 1"
epoch_folder=$(docker exec $CONTAINER_NAME /bin/bash -c "$find_epoch_cmd" | tr -d '\r')

if [[ -z "$epoch_folder" ]]; then
    echo -e "\e[31mError: No epoch folder found in $FULL_OUTPUT_DIR\e[0m"
    exit 1
else
    echo -e "\e[32mIdentified epoch folder: $epoch_folder\e[0m"
fi

merged_model_path="${FULL_OUTPUT_DIR}/merged_model"
python_command="source /opt/conda/bin/activate kolo_env && python /app/scripts/merge_lora.py --lora_model '$epoch_folder' --merged_model '$merged_model_path'"

if [[ -n "$QUANTIZATION" ]]; then
    python_command+=" --quantization '$QUANTIZATION'"
fi

echo -e "\e[33mExecuting merge command inside container '$CONTAINER_NAME':\e[0m"
echo -e "\e[33m$python_command\e[0m"

if ! docker exec -it $CONTAINER_NAME /bin/bash -c "$python_command"; then
    echo -e "\e[31mFailed to execute merge script.\e[0m"
    exit 1
else
    echo -e "\e[32mMerge script executed successfully!\e[0m"
fi

conversion_command="source /opt/conda/bin/activate kolo_env && /app/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile '$FULL_OUTPUT_DIR/Merged.gguf' '$merged_model_path'"
echo -e "\e[33mExecuting conversion command inside container '$CONTAINER_NAME':\e[0m"
echo -e "\e[33m$conversion_command\e[0m"

if ! docker exec -it $CONTAINER_NAME /bin/bash -c "$conversion_command"; then
    echo -e "\e[31mFailed to execute conversion script.\e[0m"
    exit 1
else
    echo -e "\e[32mConversion script executed successfully!\e[0m"
fi

# --- Begin quantization step ---
if [[ -z "$QUANTIZATION" ]]; then
    echo -e "\e[33mQuantization parameter not provided. Skipping quantization step.\e[0m"
else
    quant_upper=$(echo "$QUANTIZATION" | tr '[:lower:]' '[:upper:]')
    quantize_command="source /opt/conda/bin/activate kolo_env && /app/llama.cpp/llama-quantize '$FULL_OUTPUT_DIR/Merged.gguf' '$FULL_OUTPUT_DIR/Merged${QUANTIZATION}.gguf' $quant_upper"
    echo -e "\e[33mExecuting quantization command inside container '$CONTAINER_NAME':\e[0m"
    echo -e "\e[33m$quantize_command\e[0m"

    if ! docker exec -it $CONTAINER_NAME /bin/bash -c "$quantize_command"; then
        echo -e "\e[31mFailed to execute quantization script.\e[0m"
        exit 1
    else
        echo -e "\e[32mQuantization script executed successfully!\e[0m"
    fi
fi