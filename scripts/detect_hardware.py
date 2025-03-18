#!/usr/bin/env python3
import subprocess
import os

def get_gpu_info():
    try:
        lspci_output = subprocess.check_output("lspci | grep -i vga", shell=True).decode("utf-8")
        if "nvidia" in lspci_output.lower():
            return "nvidia"
        elif "amd" in lspci_output.lower() or "radeon" in lspci_output.lower():
            return "amd"
        else:
            return "cpu"
    except:
        return "cpu"

with open("/tmp/hardware_type.txt", "w") as f:
    f.write(get_gpu_info())