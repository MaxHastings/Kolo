#!/bin/bash

# Stop script on error
set -e

# Run the container
echo "Running Docker container..."
docker volume create kolo_volume
docker run -p 2222:22 -p 8080:8080 -v kolo_volume:/var/kolo_data -it -d --name kolo_container kolo