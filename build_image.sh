#!/bin/bash

# Stop script on error
set -e

# Default hardware type
HARDWARE_TYPE="cpu"

# Function to display usage
show_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Build the Kolo Docker image with specific hardware configuration."
  echo ""
  echo "Options:"
  echo "  -h, --help             Show this help message"
  echo "  -t, --type TYPE        Specify hardware type: 'cpu', 'cuda' (for NVIDIA), or 'rocm' (for AMD)"
  echo ""
  echo "Example:"
  echo "  $0 --type cuda         # Build with NVIDIA CUDA support"
  echo "  $0 --type rocm         # Build with AMD ROCm support"
  echo "  $0 --type cpu          # Build for CPU only (default)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--type)
      HARDWARE_TYPE="$2"
      shift
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Validate hardware type
if [[ "$HARDWARE_TYPE" != "cpu" && "$HARDWARE_TYPE" != "cuda" && "$HARDWARE_TYPE" != "rocm" ]]; then
  echo "Error: Invalid hardware type. Must be 'cpu', 'cuda', or 'rocm'."
  show_usage
  exit 1
fi

echo "Building Docker image for hardware type: $HARDWARE_TYPE"

# Build the Docker image with hardware type as a build argument
docker build --build-arg HARDWARE_TYPE="$HARDWARE_TYPE" -t kolo .

echo "Docker image built successfully!"
echo "Hardware configuration: $HARDWARE_TYPE"
echo ""
echo "To run the container:"
echo "docker run -it --name kolo-container -p 8080:8080 -p 2222:22 kolo"