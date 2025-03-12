import os
import re
import yaml
import argparse
import requests
import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from SyntheticDataGeneration.QAGenerator import QAGeneratorEngine
from SyntheticDataGeneration.Utils import Utils

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA data for LLM fine-tuning.")
    parser.add_argument("--config", default="generate_qa_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--threads", type=int, default=8, help="Max workers for processing all tasks")
    # New argument to pass in the QA generation output folder name
    parser.add_argument("--qa_output", default="qa_generation_output", help="QA generation output folder name")
    # New argument to override the model in config
    parser.add_argument("--model", default=None, help="Override model for both question and answer providers")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        Utils.logger.error(f"Configuration file not found: {config_path}")
        return

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_base_path = Path(config.get("global", {}).get("output_base_path", "/var/kolo_data"))
    # Pass the qa_output and model arguments to the engine
    engine = QAGeneratorEngine(config, output_base_path, args.threads, qa_output=args.qa_output, model_override=args.model)
    engine.run()

if __name__ == "__main__":
    main()
