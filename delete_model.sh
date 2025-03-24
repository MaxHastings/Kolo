#!/bin/bash

# Usage Example:
# ./delete_model.sh "GodOutput" "unsloth"

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments."
    echo "Usage: ./delete_model.sh <DirFolder> <Tool>"
    exit 1
fi

# Get arguments
DIR_FOLDER=$1  # The subdirectory to remove under the tool folder
TOOL=$2        # The tool directory (either "unsloth" or "torchtune")

# Validate Tool argument
if [ "$TOOL" != "unsloth" ] && [ "$TOOL" != "torchtune" ]; then
    echo "Error: Tool must be either 'unsloth' or 'torchtune'."
    exit 1
fi

# Define container name
CONTAINER_NAME="kolo_container"

# Full path used for container operations
FULL_PATH="/var/kolo_data/$TOOL/$DIR_FOLDER"

# Confirmation path that the user must type
CONFIRM_PATH="/$TOOL/$DIR_FOLDER"

# Check if the container is running
if ! docker ps --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    exit 1
fi

# Check if the directory exists inside the container
DIR_CHECK=$(docker exec -it $CONTAINER_NAME sh -c "if [ -d '$FULL_PATH' ]; then echo 'exists'; else echo 'not_exists'; fi")
if [ "$DIR_CHECK" = "not_exists" ]; then
    echo "Error: Directory '$FULL_PATH' does not exist inside container '$CONTAINER_NAME'."
    exit 1
fi

# Inform the user exactly what to type to confirm deletion
echo "WARNING: You are about to permanently delete the directory '$FULL_PATH' inside container '$CONTAINER_NAME'."
echo "To confirm deletion, you MUST type EXACTLY the following directory path:"
echo -e "\t$CONFIRM_PATH"
read -p "Type the directory path to confirm deletion: " CONFIRMATION

if [ "$CONFIRMATION" != "$CONFIRM_PATH" ]; then
    echo "Error: Confirmation failed. The text you entered does not match '$CONFIRM_PATH'. Aborting."
    exit 1
fi

# Execute the remove command inside the container
echo "Deleting '$FULL_PATH' inside container '$CONTAINER_NAME'..."
if docker exec -it $CONTAINER_NAME rm -rf "$FULL_PATH"; then
    echo "Directory '$FULL_PATH' removed successfully!"
else
    echo "Failed to remove directory '$FULL_PATH'."
    exit 1
fi