#!/bin/bash

# Default values
baseModel="Meta-llama/Llama-3.2-1B-Instruct"
outputDir="outputs"
quantization="Q4_K_M"
useCheckpoint=false
fastTransfer=false
gpuArch=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --learningRate)
      learningRate="$2"
      shift 2
      ;;
    --trainData)
      trainData="$2"
      shift 2
      ;;
    --baseModel)
      baseModel="$2"
      shift 2
      ;;
    --loraRank)
      loraRank="$2"
      shift 2
      ;;
    --loraAlpha)
      loraAlpha="$2"
      shift 2
      ;;
    --loraDropout)
      loraDropout="$2"
      shift 2
      ;;
    --maxSeqLength)
      maxSeqLength="$2"
      shift 2
      ;;
    --warmupSteps)
      warmupSteps="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --schedulerType)
      schedulerType="$2"
      shift 2
      ;;
    --batchSize)
      batchSize="$2"
      shift 2
      ;;
    --outputDir)
      outputDir="$2"
      shift 2
      ;;
    --quantization)
      quantization="$2"
      shift 2
      ;;
    --weightDecay)
      weightDecay="$2"
      shift 2
      ;;
    --useCheckpoint)
      useCheckpoint=true
      shift
      ;;
    --hfToken)
      hfToken="$2"
      shift 2
      ;;
    --fastTransfer)
      fastTransfer=true
      shift
      ;;
    --gpuArch)
      gpuArch="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Log received parameters
echo -e "\e[36mParameters passed:\e[0m"
[ ! -z "$epochs" ] && echo "Epochs: $epochs"
[ ! -z "$learningRate" ] && echo "LearningRate: $learningRate"
[ ! -z "$trainData" ] && echo "TrainData: $trainData"
[ ! -z "$baseModel" ] && echo "BaseModel: $baseModel"
[ ! -z "$loraRank" ] && echo "LoraRank: $loraRank"
[ ! -z "$loraAlpha" ] && echo "LoraAlpha: $loraAlpha"
[ ! -z "$loraDropout" ] && echo "LoraDropout: $loraDropout"
[ ! -z "$maxSeqLength" ] && echo "MaxSeqLength: $maxSeqLength"
[ ! -z "$warmupSteps" ] && echo "WarmupSteps: $warmupSteps"
[ ! -z "$seed" ] && echo "Seed: $seed"
[ ! -z "$schedulerType" ] && echo "SchedulerType: $schedulerType"
[ ! -z "$batchSize" ] && echo "BatchSize: $batchSize"
[ ! -z "$outputDir" ] && echo "OutputDir: $outputDir"
[ ! -z "$quantization" ] && echo "Quantization: $quantization"
[ ! -z "$weightDecay" ] && echo "WeightDecay: $weightDecay"
[ "$useCheckpoint" = true ] && echo "UseCheckpoint: Enabled" || echo "UseCheckpoint: Disabled"

# Log GPU mode or fast transfer based on parameters
if [ ! -z "$gpuArch" ]; then
  echo -e "\e[36mGPU Architecture: $gpuArch\e[0m"
else
  if [ "$fastTransfer" = true ]; then
    echo -e "\e[36mFastTransfer: Enabled (HF_HUB_ENABLE_HF_TRANSFER=1)\e[0m"
  else
    echo -e "\e[36mFastTransfer: Disabled (HF_HUB_ENABLE_HF_TRANSFER=0)\e[0m"
  fi
fi

# Define the Docker container name and check if it is running
containerName="kolo_container"
if ! docker ps --format "{{.Names}}" | grep -q "$containerName"; then
  echo -e "\e[31mError: Container '$containerName' is not running.\e[0m"
  exit 1
fi

# Define BaseModel to config mapping
declare -A configMap
configMap["Meta-llama/Llama-3.1-8B-Instruct"]="/app/torchtune/configs/llama3_1/8B_qlora_single_device.yaml"
configMap["Meta-llama/Llama-3.2-3B-Instruct"]="/app/torchtune/configs/llama3_2/3B_qlora_single_device.yaml"
configMap["Meta-llama/Llama-3.2-1B-Instruct"]="/app/torchtune/configs/llama3_2/1B_qlora_single_device.yaml"

# Retrieve the configuration value based on the provided BaseModel
if [ "${configMap[$baseModel]+isset}" ]; then
  configValue="${configMap[$baseModel]}"
else
  echo -e "\e[31mError: The specified BaseModel '$baseModel' was not found in the configuration mapping.\e[0m"
  exit 1
fi

echo -e "\e[36mUsing configuration: $configValue for BaseModel: $baseModel\e[0m"

# Begin BaseModel download step
if [ -z "$hfToken" ]; then
  echo -e "\e[31mError: Hugging Face token must be provided.\e[0m"
  exit 1
fi

hfTransferValue=0
[ "$fastTransfer" = true ] && hfTransferValue=1

downloadCommand="export HF_HUB_ENABLE_HF_TRANSFER=$hfTransferValue && source /opt/conda/bin/activate kolo_env && tune download $baseModel --ignore-patterns 'original/consolidated.00.pth' --hf-token '$hfToken'"

echo -e "\e[33mDownloading BaseModel using command:\e[0m"
echo -e "\e[33m$downloadCommand\e[0m"

if ! docker exec -it $containerName /bin/bash -c "$downloadCommand"; then
  echo -e "\e[31mFailed to download BaseModel.\e[0m"
  exit 1
else
  echo -e "\e[32mBaseModel downloaded successfully!\e[0m"
fi

# Begin torchtune run
# Build the base torchtune command string using the configuration from the mapping
if [ ! -z "$gpuArch" ]; then
  # AMD GPU branch: set HIP alloc conf and ROCm arch
  command="export PYTORCH_HIP_ALLOC_CONF='garbage_collection_threshold:0.8,max_split_size_mb:512' && PYTORCH_ROCM_ARCH=$gpuArch source /opt/conda/bin/activate kolo_env && tune run lora_finetune_single_device --config $configValue"
else
  # Default branch
  command="source /opt/conda/bin/activate kolo_env && tune run lora_finetune_single_device --config $configValue"
fi

# Append dynamic parameters with defaults
if [ ! -z "$epochs" ]; then
  command+=" epochs=$epochs"
else
  command+=" epochs=3"
fi

if [ ! -z "$batchSize" ]; then
  command+=" batch_size=$batchSize"
else
  command+=" batch_size=1"
fi

if [ ! -z "$trainData" ]; then
  command+=" dataset.data_files='$trainData'"
else
  command+=" dataset.data_files=./data.json"
fi

# Fixed dataset parameters
command+=" dataset._component_=torchtune.datasets.chat_dataset"
command+=" dataset.source=json"
command+=" dataset.conversation_column=conversations"
command+=" dataset.conversation_style=sharegpt"

if [ ! -z "$loraRank" ]; then
  command+=" model.lora_rank=$loraRank"
else
  command+=" model.lora_rank=16"
fi

if [ ! -z "$loraAlpha" ]; then
  command+=" model.lora_alpha=$loraAlpha"
else
  command+=" model.lora_alpha=16"
fi

[ ! -z "$loraDropout" ] && command+=" model.lora_dropout=$loraDropout"

if [ ! -z "$learningRate" ]; then
  command+=" optimizer.lr=$learningRate"
else
  command+=" optimizer.lr=1e-4"
fi

[ ! -z "$maxSeqLength" ] && command+=" tokenizer.max_seq_len=$maxSeqLength"

if [ ! -z "$warmupSteps" ]; then
  command+=" lr_scheduler.num_warmup_steps=$warmupSteps"
else
  command+=" lr_scheduler.num_warmup_steps=100"
fi

[ ! -z "$seed" ] && command+=" seed=$seed"

if [ ! -z "$schedulerType" ]; then
  command+=" lr_scheduler._component_=torchtune.training.lr_schedulers.get_${schedulerType}_schedule_with_warmup"
else
  command+=" lr_scheduler._component_=torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup"
fi

if [ ! -z "$weightDecay" ]; then
  command+=" optimizer.weight_decay=$weightDecay"
else
  command+=" optimizer.weight_decay=0.01"
fi

if [ "$useCheckpoint" = true ]; then
  command+=" resume_from_checkpoint=True"
else
  command+=" resume_from_checkpoint=False"
fi

# Set the output directory; default is "outputs"
fullOutputDir="/var/kolo_data/torchtune/$outputDir"
command+=" output_dir='$fullOutputDir'"

# Log a note on quantization if provided
[ ! -z "$quantization" ] && echo "Note: Quantization parameter '$quantization' is provided and will be used for quantization."

echo -e "\e[33mExecuting torchtune command inside container '$containerName':\e[0m"
echo -e "\e[33m$command\e[0m"

if ! docker exec -it $containerName /bin/bash -c "$command"; then
  echo -e "\e[31mFailed to execute torchtune run.\e[0m"
  exit 1
else
  echo -e "\e[32mTorchtune run completed successfully!\e[0m"
fi

# Begin post-run merging steps
findEpochCmd="ls -d ${fullOutputDir}/epoch_* 2>/dev/null | sort -V | tail -n 1"
epochFolder=$(docker exec $containerName /bin/bash -c "$findEpochCmd" | tr -d '\r')

if [ -z "$epochFolder" ]; then
  echo -e "\e[31mError: No epoch folder found in $fullOutputDir\e[0m"
  exit 1
else
  echo -e "\e[32mIdentified epoch folder: $epochFolder\e[0m"
fi

mergedModelPath="${fullOutputDir}/merged_model"
pythonCommand="source /opt/conda/bin/activate kolo_env && python /app/merge_lora.py --lora_model '$epochFolder' --merged_model '$mergedModelPath'"
[ ! -z "$quantization" ] && pythonCommand+=" --quantization '$quantization'"

echo -e "\e[33mExecuting merge command inside container '$containerName':\e[0m"
echo -e "\e[33m$pythonCommand\e[0m"

if ! docker exec -it $containerName /bin/bash -c "$pythonCommand"; then
  echo -e "\e[31mFailed to execute merge script.\e[0m"
  exit 1
else
  echo -e "\e[32mMerge script executed successfully!\e[0m"
fi

conversionCommand="source /opt/conda/bin/activate kolo_env && /app/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile '$fullOutputDir/Merged.gguf' '$mergedModelPath'"
echo -e "\e[33mExecuting conversion command inside container '$containerName':\e[0m"
echo -e "\e[33m$conversionCommand\e[0m"

if ! docker exec -it $containerName /bin/bash -c "$conversionCommand"; then
  echo -e "\e[31mFailed to execute conversion script.\e[0m"
  exit 1
else
  echo -e "\e[32mConversion script executed successfully!\e[0m"
fi

# Begin quantization step
if [ -z "$quantization" ]; then
  echo -e "\e[33mQuantization parameter not provided. Skipping quantization step.\e[0m"
else
  quantUpper=$(echo "$quantization" | tr '[:lower:]' '[:upper:]')
  quantizeCommand="source /opt/conda/bin/activate kolo_env && /app/llama.cpp/llama-quantize '$fullOutputDir/Merged.gguf' '$fullOutputDir/Merged${quantization}.gguf' $quantUpper"
  echo -e "\e[33mExecuting quantization command inside container '$containerName':\e[0m"
  echo -e "\e[33m$quantizeCommand\e[0m"

  if ! docker exec -it $containerName /bin/bash -c "$quantizeCommand"; then
    echo -e "\e[31mFailed to execute quantization script.\e[0m"
    exit 1
  else
    echo -e "\e[32mQuantization script executed successfully!\e[0m"
  fi
fi