import os
import yaml
import argparse
import requests
import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from SyntheticDataGeneration.ApiClient import APIClient
from SyntheticDataGeneration.FileManager import FileManager
from SyntheticDataGeneration.Utils import Utils

class FileGroupProcessor:
    def __init__(
        self,
        group_name: str,
        group_config: Dict[str, Any],
        config: Dict[str, Any],
        full_base_dir: Path,
        output_base_path: Path,
        qa_output: str,
        answer_api_client: APIClient,
        thread_count: int,
        file_manager: FileManager
    ):
        self.group_name = group_name
        self.group_config = group_config
        self.config = config
        self.full_base_dir = full_base_dir
        self.output_base_path = output_base_path
        self.qa_output = qa_output
        self.answer_api_client = answer_api_client
        self.thread_count = thread_count
        self.file_manager = file_manager

        # Load configuration sections
        self.file_headers = config.get("FileHeaders", [])
        self.answer_instruction_lists = config.get("AnswerInstructionList", [])
        self.answer_prompts = config.get("AnswerPrompt", [])

        # Prepare output directories
        self.questions_dir = self.output_base_path / self.qa_output / "questions"  # You might not use this now
        self.answers_dir = self.output_base_path / self.qa_output / "answers"
        self.debug_dir = self.output_base_path / self.qa_output / "debug"
        for d in [self.questions_dir, self.answers_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_templates(self) -> bool:
        file_header_name = self.group_config.get("file_header", "")
        file_header_obj = Utils.get_item_by_name(self.file_headers, file_header_name)
        self.file_header_template = file_header_obj["description"] if file_header_obj else ""
        
        answer_prompt_name = self.group_config.get("answer_prompt", "")
        answer_prompt_obj = Utils.get_item_by_name(self.answer_prompts, answer_prompt_name)
        if not answer_prompt_obj:
            Utils.logger.error(f"No answer prompt found for name '{answer_prompt_name}'.")
            return False
        self.answer_prompt_template = answer_prompt_obj["description"]
        return True

    def generate_file_content(self, file_list: List[str], for_questions: bool = True) -> str:
        return self.file_manager.build_files_content(file_list, self.file_header_template)

    def generate_answer(
        self, question_number: int, question_text: str, answer_instruction: str, combined_content: str
    ):
        final_prompt = self.answer_prompt_template.format(
            file_content=combined_content,
            instruction=answer_instruction,
            question=question_text
        )
        ans_instr_hash = Utils.get_hash(answer_instruction)[:8]
        answer_filename = f"answer_{self.group_name}_q{question_number}_{ans_instr_hash}.txt"
        debug_filename = f"debug_{self.group_name}_q{question_number}_{ans_instr_hash}.txt"
        meta_filename = f"answer_{self.group_name}_q{question_number}_{ans_instr_hash}.meta"

        answer_file_path = self.answers_dir / answer_filename
        answer_debug_path = self.debug_dir / debug_filename
        meta_file_path = self.answers_dir / meta_filename

        current_hash = Utils.get_hash(final_prompt)
        regenerate = True
        if answer_file_path.exists():
            if meta_file_path.exists():
                stored_hash = self.file_manager.read_text(meta_file_path).strip()
                if stored_hash == current_hash:
                    Utils.logger.info(
                        f"[Group: {self.group_name}] Answer for question {question_number} is up to date."
                    )
                    regenerate = False
                else:
                    Utils.logger.info(f"[Group: {self.group_name}] Changed prompt detected, regenerating answer for question {question_number}.")
            else:
                self.file_manager.write_text(meta_file_path, current_hash)
                regenerate = False

        if not regenerate:
            return

        answer_text = self.answer_api_client.call_api(final_prompt)
        if not answer_text:
            Utils.logger.error(
                f"[Group: {self.group_name}] Failed to generate answer for question {question_number}."
            )
            return
        self.file_manager.write_text(answer_file_path, answer_text)
        self.file_manager.write_text(answer_debug_path, final_prompt)
        self.file_manager.write_text(meta_file_path, current_hash)
        Utils.logger.info(f"[Group: {self.group_name}] Saved answer -> {answer_file_path}")

    def process(self):
        if not self.resolve_templates():
            return

        # Collect answer instructions from the configured list
        self.all_answer_instructions = []
        answer_instruction_list_names = self.group_config.get("answer_instruction_list", [])
        for a_list_name in answer_instruction_list_names:
            instr_list_obj = Utils.get_item_by_name(self.answer_instruction_lists, a_list_name)
            if instr_list_obj:
                self.all_answer_instructions.extend(instr_list_obj.get("instruction", []))

        file_list = self.group_config.get("files", [])
        combined_content = self.generate_file_content(file_list, for_questions=False)

        # Instead of generating questions, use the provided question_list from config
        questions = self.group_config.get("question_list", [])
        if not questions:
            Utils.logger.warning(f"[Group: {self.group_name}] No questions found.")
            return

        # Prepare answer tasks for each question and each answer instruction
        answer_tasks = []
        for q_num, question_text in enumerate(questions, start=1):
            for answer_instruction in self.all_answer_instructions:
                answer_tasks.append((q_num, question_text, answer_instruction))

        def handle_answer(task):
            q_num, question_text, answer_instruction = task
            self.generate_answer(q_num, question_text, answer_instruction, combined_content)

        inner_workers = self.thread_count if self.thread_count > 1 else 1
        if inner_workers > 1:
            with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                futures = [pool.submit(handle_answer, t) for t in answer_tasks]
                for f in as_completed(futures):
                    f.result()
        else:
            for t in answer_tasks:
                handle_answer(t)
