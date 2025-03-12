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

from SyntheticDataGeneration.Utils import Utils

# Try importing the OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    Utils.logger.warning("OpenAI package not installed; openai provider will not work.")

class APIClient:
    def __init__(self, provider: str, model: str, global_ollama_url: Optional[str] = None, openai_client: Optional[OpenAI] = None):
        self.provider = provider.lower()
        self.model = model
        self.global_ollama_url = global_ollama_url
        self.openai_client = openai_client

    def call_api(self, prompt: str) -> Optional[str]:
        max_retries = 5
        backoff_factor = 1
        if self.provider == "openai":
            if not self.openai_client:
                Utils.logger.error("OpenAI client not initialized.")
                return None
            attempt = 0
            while attempt <= max_retries:
                try:
                    response = self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    Utils.logger.error(f"OpenAI API error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries:
                        return None
                    sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    Utils.logger.info(f"Retrying OpenAI API call in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    attempt += 1
        elif self.provider == "ollama":
            if not self.global_ollama_url:
                Utils.logger.error("Global Ollama URL not provided.")
                return None
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {}
            }
            attempt = 0
            while attempt <= max_retries:
                try:
                    response = requests.post(self.global_ollama_url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    return result.get("response", "").strip()
                except Exception as e:
                    Utils.logger.error(f"Ollama API error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries:
                        return None
                    sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    Utils.logger.info(f"Retrying Ollama API call in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    attempt += 1
        else:
            Utils.logger.error(f"Unknown provider specified: {self.provider}")
            return None