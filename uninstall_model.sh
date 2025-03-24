#!/bin/bash

# Usage Example:
# ./uninstall_model.sh "my_model_name"

# Check if model name is provided
if [ $# -lt 1 ]; then
    echo "Error: Model name is required."
    echo "Usage: ./uninstall_model.sh \"model_name\""
    exit 1
fi

# Get the model name from the first argument
MODEL_NAME="$1"

# Define container name
CONTAINER_NAME="kolo_container"

# Check if the container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' is not running." >&2
    exit 1
fi

# Execute the Ollama remove command inside the container
echo "Removing Ollama model '$MODEL_NAME' inside container '$CONTAINER_NAME'..."
if docker exec -it $CONTAINER_NAME ollama rm "$MODEL_NAME"; then
    echo "Ollama model '$MODEL_NAME' removed successfully!"
else
    echo "Failed to remove Ollama model." >&2
    exit 1
fi