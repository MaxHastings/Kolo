#!/bin/bash
python3 /tmp/detect_hardware.py
HARDWARE_TYPE=$(cat /tmp/hardware_type.txt)
echo "Running with detected hardware: $HARDWARE_TYPE"
exec "$@"