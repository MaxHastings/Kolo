#!/bin/bash

# Stop script on error
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t kolo .
