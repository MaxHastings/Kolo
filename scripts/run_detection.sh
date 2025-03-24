#!/bin/bash

# Get hardware type from file or environment variable
if [ -f /tmp/hardware_type.txt ]; then
    HARDWARE_TYPE=$(cat /tmp/hardware_type.txt)
else
    HARDWARE_TYPE="${HARDWARE_TYPE:-cpu}"
fi

echo "Running container with hardware type: $HARDWARE_TYPE"

# Additional setup specific to each hardware type at runtime
case "$HARDWARE_TYPE" in
  "cuda")
    echo "Setting up NVIDIA runtime environment..."
    # Check if NVIDIA drivers are available
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi || echo "Warning: NVIDIA drivers detected but nvidia-smi failed. Check driver installation."
    else
        echo "Warning: NVIDIA drivers not detected. CUDA may not work properly."
    fi
    ;;
    
  "rocm")
    echo "Setting up AMD ROCm runtime environment..."
    # Check if ROCm is available
    if command -v rocm-smi &> /dev/null; then
        rocm-smi || echo "Warning: ROCm detected but rocm-smi failed. Check ROCm installation."
    else
        echo "Warning: ROCm not detected. ROCm features may not work properly."
    fi
    ;;
    
  "cpu")
    echo "Using CPU-only configuration."
    ;;
    
  *)
    echo "Unknown hardware type: $HARDWARE_TYPE"
    echo "Defaulting to CPU configuration."
    ;;
esac

# Execute CMD or passed command
exec "$@"