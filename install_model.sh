#!/bin/bash

# Usage Example:
# ./install_model.sh "God" -t "unsloth" -o "GodOutput" -q "Q4_K_M"
# ./install_model.sh "God" -t "torchtune" -o "GodOutput" -q "Q4_K_M"

# Default values
MODEL_NAME=""
OUTPUT_DIR=""
QUANTIZATION=""
TOOL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tool)
            TOOL="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        *)
            MODEL_NAME="$1"
            shift
            ;;
    esac
done

# Validate required parameters
if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: Model name is required."
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory (-o) is required."
    exit 1
fi

if [[ -z "$QUANTIZATION" ]]; then
    echo "Error: Quantization (-q) is required."
    exit 1
fi

if [[ -z "$TOOL" ]]; then
    echo "Error: Tool (-t) is required."
    exit 1
fi

# Validate tool option
if [[ "$TOOL" != "torchtune" && "$TOOL" != "unsloth" ]]; then
    echo "Error: Tool must be either 'torchtune' or 'unsloth'."
    exit 1
fi

# Define the container name
CONTAINER_NAME="kolo_container"

# Check if the container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' is not running." >&2
    exit 1
fi

# Construct the full path to the model file using the chosen data source
BASE_DIR="/var/kolo_data/$TOOL"
MODEL_FILE_PATH="$BASE_DIR/$OUTPUT_DIR/Modelfile$QUANTIZATION"

# Execute the Ollama command inside the container
echo "Creating Ollama model '$MODEL_NAME' using file '$MODEL_FILE_PATH' inside container '$CONTAINER_NAME'..."
if docker exec -it $CONTAINER_NAME ollama create "$MODEL_NAME" -f "$MODEL_FILE_PATH"; then
    echo "Ollama model '$MODEL_NAME' created successfully!"
else
    echo "Failed to create Ollama model." >&2
    exit 1
fi