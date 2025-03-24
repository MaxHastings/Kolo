#!/bin/bash

# Define the container name
containerName="kolo_container"

# Define the target directories inside the container
targetDirectories=(
    "/var/kolo_data/torchtune"
    "/var/kolo_data/unsloth"
)

for dir in "${targetDirectories[@]}"; do
    echo -e "\e[36mModel folders in $dir\e[0m"
    # Build the command to allow shell globbing and suppress error messages if no folders are found.
    cmd="ls -d $dir/*/ 2>/dev/null"  # ls will not print an error if no folder exists
    
    # Execute the command and capture the output.
    result=$(docker exec "$containerName" sh -c "$cmd" 2>&1)
    
    # If the trimmed output is empty, no folders were found.
    if [ -z "$(echo "$result" | tr -d '[:space:]')" ]; then
        echo -e "\e[32mNo models found\e[0m"
    else
        echo -e "\e[32m$result\e[0m"
    fi
    
    echo -e "\e[33m-------------------------------------\e[0m"
done

# Now, list the installed models in Ollama using the 'ollama list' command inside the container.
echo -e "\n\e[36mListing installed models in Ollama:\e[0m"

# Build the docker exec command to run 'ollama list' inside the container.
cmd="ollama list"

# Execute the command and capture the output.
ollamaOutput=$(docker exec "$containerName" sh -c "$cmd" 2>&1)

# Check if any output was returned.
if [ -z "$(echo "$ollamaOutput" | tr -d '[:space:]')" ]; then
    echo -e "\e[32mNo models installed or no output from ollama list.\e[0m"
else
    echo -e "\e[32m$ollamaOutput\e[0m"
fi