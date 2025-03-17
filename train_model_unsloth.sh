#!/bin/bash

### Bash Script to Execute a Python Script inside a Docker Container
###
### Usage:
### ./train_model_unsloth.sh -e 3 -l 1e-4 -t "data.jsonl" -b "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" -c "llama-3.1" -r 16 -a 16 -d 0 -m 1024 -w 10 -s 500 -i 5 -S 1337 -T "linear" -B 2 -o "GodOutput" -q "Q4_K_M" -W 0 -u -f
### When -f is used, HF_HUB_ENABLE_HF_TRANSFER=1 will be set, forcing HF to use the faster but much less reliable Rust downloader (recommended only for 1Gbps+ connections)

# Default values
EPOCHS=""
LEARNING_RATE=""
TRAIN_DATA=""
BASE_MODEL=""
CHAT_TEMPLATE=""
LORA_RANK=""
LORA_ALPHA=""
LORA_DROPOUT=""
MAX_SEQ_LENGTH=""
WARMUP_STEPS=""
SAVE_STEPS=""
SAVE_TOTAL_LIMIT=""
SEED=""
SCHEDULER_TYPE=""
BATCH_SIZE=""
OUTPUT_DIR=""
QUANTIZATION=""
WEIGHT_DECAY=""
USE_CHECKPOINT=false
FAST_TRANSFER=false

# Parse command line arguments
while getopts "e:l:t:b:c:r:a:d:m:w:s:i:S:T:B:o:q:W:uf" opt; do
  case $opt in
    e) EPOCHS="$OPTARG" ;;
    l) LEARNING_RATE="$OPTARG" ;;
    t) TRAIN_DATA="$OPTARG" ;;
    b) BASE_MODEL="$OPTARG" ;;
    c) CHAT_TEMPLATE="$OPTARG" ;;
    r) LORA_RANK="$OPTARG" ;;
    a) LORA_ALPHA="$OPTARG" ;;
    d) LORA_DROPOUT="$OPTARG" ;;
    m) MAX_SEQ_LENGTH="$OPTARG" ;;
    w) WARMUP_STEPS="$OPTARG" ;;
    s) SAVE_STEPS="$OPTARG" ;;
    i) SAVE_TOTAL_LIMIT="$OPTARG" ;;
    S) SEED="$OPTARG" ;;
    T) SCHEDULER_TYPE="$OPTARG" ;;
    B) BATCH_SIZE="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    q) QUANTIZATION="$OPTARG" ;;
    W) WEIGHT_DECAY="$OPTARG" ;;
    u) USE_CHECKPOINT=true ;;
    f) FAST_TRANSFER=true ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "Parameters passed to the script:"

[ -n "$EPOCHS" ] && echo "Epochs: $EPOCHS"
[ -n "$LEARNING_RATE" ] && echo "LearningRate: $LEARNING_RATE"
[ -n "$TRAIN_DATA" ] && echo "TrainData: $TRAIN_DATA"
[ -n "$BASE_MODEL" ] && echo "BaseModel: $BASE_MODEL"
[ -n "$CHAT_TEMPLATE" ] && echo "ChatTemplate: $CHAT_TEMPLATE"
[ -n "$LORA_RANK" ] && echo "LoraRank: $LORA_RANK"
[ -n "$LORA_ALPHA" ] && echo "LoraAlpha: $LORA_ALPHA"
[ -n "$LORA_DROPOUT" ] && echo "LoraDropout: $LORA_DROPOUT"
[ -n "$MAX_SEQ_LENGTH" ] && echo "MaxSeqLength: $MAX_SEQ_LENGTH"
[ -n "$WARMUP_STEPS" ] && echo "WarmupSteps: $WARMUP_STEPS"
[ -n "$SAVE_STEPS" ] && echo "SaveSteps: $SAVE_STEPS"
[ -n "$SAVE_TOTAL_LIMIT" ] && echo "SaveTotalLimit: $SAVE_TOTAL_LIMIT"
[ -n "$SEED" ] && echo "Seed: $SEED"
[ -n "$SCHEDULER_TYPE" ] && echo "SchedulerType: $SCHEDULER_TYPE"
[ -n "$BATCH_SIZE" ] && echo "BatchSize: $BATCH_SIZE"
[ -n "$OUTPUT_DIR" ] && echo "OutputDir: $OUTPUT_DIR"
[ -n "$QUANTIZATION" ] && echo "Quantization: $QUANTIZATION"
[ -n "$WEIGHT_DECAY" ] && echo "WeightDecay: $WEIGHT_DECAY"
[ "$USE_CHECKPOINT" = true ] && echo "UseCheckpoint: Enabled" || echo "UseCheckpoint: Disabled"
[ "$FAST_TRANSFER" = true ] && echo "FastTransfer: Enabled (HF_HUB_ENABLE_HF_TRANSFER=1)" || echo "FastTransfer: Disabled (HF_HUB_ENABLE_HF_TRANSFER=0)"

# Define container name
CONTAINER_NAME="kolo_container"

# Check if the container is running
if ! docker ps --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    exit 1
fi

# Build command string dynamically
HF_TRANSFER_VALUE=$([ "$FAST_TRANSFER" = true ] && echo "1" || echo "0")
COMMAND="export HF_HUB_ENABLE_HF_TRANSFER=$HF_TRANSFER_VALUE && source /opt/conda/bin/activate kolo_env && python /app/scripts/train.py"

[ -n "$EPOCHS" ] && COMMAND+=" --epochs $EPOCHS"
[ -n "$LEARNING_RATE" ] && COMMAND+=" --learning_rate $LEARNING_RATE"
[ -n "$TRAIN_DATA" ] && COMMAND+=" --train_data '$TRAIN_DATA'"
[ -n "$BASE_MODEL" ] && COMMAND+=" --base_model '$BASE_MODEL'"
[ -n "$CHAT_TEMPLATE" ] && COMMAND+=" --chat_template '$CHAT_TEMPLATE'"
[ -n "$LORA_RANK" ] && COMMAND+=" --lora_rank $LORA_RANK"
[ -n "$LORA_ALPHA" ] && COMMAND+=" --lora_alpha $LORA_ALPHA"
[ -n "$LORA_DROPOUT" ] && COMMAND+=" --lora_dropout $LORA_DROPOUT"
[ -n "$MAX_SEQ_LENGTH" ] && COMMAND+=" --max_seq_length $MAX_SEQ_LENGTH"
[ -n "$WARMUP_STEPS" ] && COMMAND+=" --warmup_steps $WARMUP_STEPS"
[ -n "$SAVE_STEPS" ] && COMMAND+=" --save_steps $SAVE_STEPS"
[ -n "$SAVE_TOTAL_LIMIT" ] && COMMAND+=" --save_total_limit $SAVE_TOTAL_LIMIT"
[ -n "$SEED" ] && COMMAND+=" --seed $SEED"
[ -n "$SCHEDULER_TYPE" ] && COMMAND+=" --scheduler_type '$SCHEDULER_TYPE'"
[ -n "$BATCH_SIZE" ] && COMMAND+=" --batch_size $BATCH_SIZE"
[ -n "$OUTPUT_DIR" ] && COMMAND+=" --output_dir '$OUTPUT_DIR'"
[ -n "$QUANTIZATION" ] && COMMAND+=" --quantization '$QUANTIZATION'"
[ -n "$WEIGHT_DECAY" ] && COMMAND+=" --weight_decay '$WEIGHT_DECAY'"
[ "$USE_CHECKPOINT" = true ] && COMMAND+=" --use_checkpoint"

# Execute the python script inside the container
echo "Executing script inside container: $CONTAINER_NAME..."
if docker exec -it $CONTAINER_NAME /bin/bash -c "$COMMAND"; then
    echo "Script executed successfully!"
else
    echo "Failed to execute script."
    exit 1
fi