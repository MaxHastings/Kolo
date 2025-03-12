import os
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from SyntheticDataGeneration.ApiClient import APIClient
from SyntheticDataGeneration.FileManager import FileManager
from SyntheticDataGeneration.FileGroupProcessor import FileGroupProcessor
from SyntheticDataGeneration.Utils import Utils

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    Utils.logger.warning("OpenAI package not installed; openai provider will not work.")

class QAGeneratorEngine:
    def __init__(self, config: Dict[str, Any], output_base_path: Path, thread_count: int, qa_output: str = "qa_generation_output", model_override: Optional[str] = None):
        self.config = config
        self.output_base_path = output_base_path
        self.thread_count = thread_count
        self.qa_output = qa_output

        global_config = config.get("global", {})
        base_dir = global_config.get("base_dir", "")
        self.full_base_dir = output_base_path / base_dir
        self.global_ollama_url = global_config.get("ollama_url", "http://localhost:11434/api/generate")
        self.file_groups_config = config.get("file_groups", {})

        # Providers configuration
        question_provider_config = config.get("providers", {}).get("question", {})
        answer_provider_config = config.get("providers", {}).get("answer", {})

        openai_client = None
        if (question_provider_config.get("provider", "").lower() == "openai" or
            answer_provider_config.get("provider", "").lower() == "openai"):
            if OpenAI is None:
                Utils.logger.error("OpenAI client cannot be initialized because the package is missing.")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
                openai_client = OpenAI(api_key=api_key)

        # If a model override is provided, use it for both providers; otherwise use config values.
        question_model = model_override if model_override is not None else question_provider_config.get("model", "")
        answer_model = model_override if model_override is not None else answer_provider_config.get("model", "")

        self.question_api_client = APIClient(
            provider=question_provider_config.get("provider", ""),
            model=question_model,
            global_ollama_url=self.global_ollama_url,
            openai_client=openai_client
        )
        self.answer_api_client = APIClient(
            provider=answer_provider_config.get("provider", ""),
            model=answer_model,
            global_ollama_url=self.global_ollama_url,
            openai_client=openai_client
        )
        self.file_manager = FileManager(self.full_base_dir)

    def expand_file_groups(self) -> Dict[str, Dict[str, Any]]:
        expanded = {}
        for group_name, g_config in self.file_groups_config.items():
            iterations = g_config.get("iterations", 1)
            for i in range(1, iterations + 1):
                key = f"{group_name}_{i}"
                expanded[key] = g_config
        return expanded

    def run(self):
        expanded_groups = self.expand_file_groups()
        total_groups = len(expanded_groups)
        Utils.logger.info(f"Starting processing of {total_groups} file groups with up to {self.thread_count} threads...")
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = []
            for group_name, group_conf in expanded_groups.items():
                # Pass qa_output to the FileGroupProcessor
                processor = FileGroupProcessor(
                    group_name=group_name,
                    group_config=group_conf,
                    config=self.config,
                    full_base_dir=self.full_base_dir,
                    output_base_path=self.output_base_path,
                    qa_output=self.qa_output,
                    question_api_client=self.question_api_client,
                    answer_api_client=self.answer_api_client,
                    thread_count=self.thread_count,
                    file_manager=self.file_manager
                )
                futures.append(executor.submit(processor.process))
            for future in as_completed(futures):
                future.result()
        Utils.logger.info("All file groups have been processed successfully.")
