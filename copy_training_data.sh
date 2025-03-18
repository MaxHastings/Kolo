#!/bin/bash

# Default values
CONTAINER_NAME="kolo_container"
DEST_FILE="data.jsonl"
JSON_OUTPUT_FILE="data.json"
SCRIPT_PATH="/home/ailab/pr/Kolo/scripts/convert_jsonl_to_json.py"  # Update path

# Display help
function show_help {
    echo "Usage: $0 -f <local_file_path> [-d <destination_filename>]"
    echo "  -f    Local file path"
    echo "  -d    Destination filename inside the container (default: data.jsonl)"
    echo "  -h    Display this help message"
    exit 0
}

# Parse command line arguments
while getopts "f:d:h" opt; do
    case $opt in
        f) LOCAL_FILE="$OPTARG" ;;
        d) DEST_FILE="$OPTARG" ;;
        h) show_help ;;
        *) echo "Invalid option: -$OPTARG" >&2; show_help ;;
    esac
done

# Check if local file path is provided
if [ -z "$LOCAL_FILE" ]; then
    echo -e "\e[31mError: Local file path must be provided using -f option.\e[0m"
    show_help
    exit 1
fi

# Check if the file exists locally
if [ ! -f "$LOCAL_FILE" ]; then
    echo -e "\e[31mError: File does not exist at path: $LOCAL_FILE\e[0m"
    exit 1
fi

# Step 1: Copy the JSONL file into the container
echo "Copying $LOCAL_FILE to container $CONTAINER_NAME at /app/$DEST_FILE..."
if ! docker cp "$LOCAL_FILE" "$CONTAINER_NAME:/app/$DEST_FILE"; then
    echo -e "\e[31mFailed to copy JSONL file.\e[0m"
    exit 1
else
    echo -e "\e[32mFile copied successfully as $DEST_FILE!\e[0m"
fi

# Step 2: Copy the conversion script to the container
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "\e[31mError: Conversion script not found at $SCRIPT_PATH\e[0m"
    exit 1
fi

echo "Copying conversion script to container..."
if ! docker cp "$SCRIPT_PATH" "$CONTAINER_NAME:/app/convert_jsonl_to_json.py"; then
    echo -e "\e[31mFailed to copy conversion script.\e[0m"
    exit 1
else
    echo -e "\e[32mConversion script copied successfully!\e[0m"
fi

# Step 3: Run the conversion script in the container
echo "Running conversion script in container $CONTAINER_NAME..."
if ! docker exec "$CONTAINER_NAME" bash -c "source /opt/conda/bin/activate kolo_env && python /app/convert_jsonl_to_json.py '/app/$DEST_FILE' '/app/$JSON_OUTPUT_FILE'"; then
    echo -e "\e[31mFailed to run conversion script.\e[0m"
    exit 1
else
    echo -e "\e[32mConversion successful! Converted file created as $JSON_OUTPUT_FILE in the container.\e[0m"
fi
