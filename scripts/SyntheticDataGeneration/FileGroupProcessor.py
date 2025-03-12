# SyntheticDataGeneration/FileGroupProcessor.py

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
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from SyntheticDataGeneration.ApiClient import APIClient
from SyntheticDataGeneration.FileManager import FileManager
from SyntheticDataGeneration.Utils import Utils
from SyntheticDataGeneration.TextParser import TextParser

class FileGroupProcessor:
    def __init__(
        self,
        group_name: str,
        group_config: Dict[str, Any],
        config: Dict[str, Any],
        full_base_dir: Path,
        output_base_path: Path,
        qa_output: str,  # New parameter for QA output folder name
        question_api_client: APIClient,
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
        self.question_api_client = question_api_client
        self.answer_api_client = answer_api_client
        self.thread_count = thread_count
        self.file_manager = file_manager

        # Extract configuration sections
        self.file_headers = config.get("FileHeaders", [])
        self.question_prompts = config.get("QuestionPrompt", [])
        self.answer_prompts = config.get("AnswerPrompt", [])
        self.question_instruction_lists = config.get("QuestionInstructionList", [])
        self.answer_instruction_lists = config.get("AnswerInstructionList", [])
        self.generate_question_lists = config.get("GenerateQuestionLists", [])

        # Prepare output directories using the passed qa_output folder name
        self.questions_dir = self.output_base_path / self.qa_output / "questions"
        self.answers_dir = self.output_base_path / self.qa_output / "answers"
        self.debug_dir = self.output_base_path / self.qa_output / "debug"
        for d in [self.questions_dir, self.answers_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_templates(self) -> bool:
        file_header_name = self.group_config.get("file_header", "")
        file_header_obj = Utils.get_item_by_name(self.file_headers, file_header_name)
        self.file_header_template = file_header_obj["description"] if file_header_obj else ""

        question_prompt_name = self.group_config.get("question_prompt", "")
        question_prompt_obj = Utils.get_item_by_name(self.question_prompts, question_prompt_name)
        if not question_prompt_obj:
            Utils.logger.error(f"No question prompt found for name '{question_prompt_name}'.")
            return False
        self.question_prompt_template = question_prompt_obj["description"]

        answer_prompt_name = self.group_config.get("answer_prompt", "")
        answer_prompt_obj = Utils.get_item_by_name(self.answer_prompts, answer_prompt_name)
        if not answer_prompt_obj:
            Utils.logger.error(f"No answer prompt found for name '{answer_prompt_name}'.")
            return False
        self.answer_prompt_template = answer_prompt_obj["description"]
        return True

    def collect_instructions_and_seeds(self):
        # Collect question instructions
        question_instruction_list_names = self.group_config.get("question_instruction_list", [])
        self.all_question_instructions = []
        for q_list_name in question_instruction_list_names:
            instr_list_obj = Utils.get_item_by_name(self.question_instruction_lists, q_list_name)
            if instr_list_obj:
                self.all_question_instructions.extend(instr_list_obj.get("instruction", []))

        # Collect answer instructions
        answer_instruction_list_names = self.group_config.get("answer_instruction_list", [])
        self.all_answer_instructions = []
        for a_list_name in answer_instruction_list_names:
            instr_list_obj = Utils.get_item_by_name(self.answer_instruction_lists, a_list_name)
            if instr_list_obj:
                self.all_answer_instructions.extend(instr_list_obj.get("instruction", []))

        # Collect question seeds
        generate_question_list_names = self.group_config.get("generate_question_list", [])
        self.all_question_seeds = []
        for gq_list_name in generate_question_list_names:
            gq_obj = Utils.get_item_by_name(self.generate_question_lists, gq_list_name)
            if gq_obj:
                self.all_question_seeds.extend(gq_obj.get("questions", []))

    def generate_file_content(self, file_list: List[str], for_questions: bool = True) -> str:
        return self.file_manager.build_files_content(file_list, self.file_header_template)

    def generate_question_task(
        self, q_seed_idx: int, instr_idx: int, seed_text: str, instruction: str, combined_content: str, file_list: List[str]
    ) -> Optional[str]:
        file_name_list_str = ", ".join(file_list)
        final_prompt = self.question_prompt_template.format(
            file_content=combined_content,
            generate_question=seed_text,
            instruction=instruction,
            file_name_list=file_name_list_str
        )
        out_filename = f"questions_{self.group_name}_seed{q_seed_idx}_instr{instr_idx}.txt"
        debug_filename = f"debug_{self.group_name}_seed{q_seed_idx}_instr{instr_idx}_questions.txt"
        questions_path = self.questions_dir / out_filename
        debug_path = self.debug_dir / debug_filename

        if questions_path.exists():
            question_text = self.file_manager.read_text(questions_path).strip()
            Utils.logger.info(f"[Group: {self.group_name}] Using existing questions file: {out_filename}")
        else:
            question_text = self.question_api_client.call_api(final_prompt)
            if not question_text:
                Utils.logger.error(f"[Group: {self.group_name}] Failed to generate questions (seed={q_seed_idx}, instr={instr_idx}).")
                return None
            self.file_manager.write_text(questions_path, question_text)
            self.file_manager.write_text(debug_path, final_prompt)
        return question_text

    def generate_answer(
        self, q_seed_idx: int, instr_idx: int, question_number: int, question_text: str,
        answer_instruction: str, combined_content: str
    ):
        final_prompt = self.answer_prompt_template.format(
            file_content=combined_content,
            instruction=answer_instruction,
            question=question_text
        )
        ans_instr_hash = Utils.get_hash(answer_instruction)[:8]
        answer_filename = f"answer_{self.group_name}_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{ans_instr_hash}.txt"
        debug_filename = f"debug_{self.group_name}_answer_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{ans_instr_hash}.txt"
        meta_filename = f"answer_{self.group_name}_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{ans_instr_hash}.meta"

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
                        f"[Group: {self.group_name}] Answer for (seed={q_seed_idx}, instr={instr_idx}, q={question_number}) is up to date."
                    )
                    regenerate = False
                else:
                    Utils.logger.info(f"[Group: {self.group_name}] Changed prompt detected, regenerating answer.")
            else:
                self.file_manager.write_text(meta_file_path, current_hash)
                regenerate = False

        if not regenerate:
            return

        answer_text = self.answer_api_client.call_api(final_prompt)
        if not answer_text:
            Utils.logger.error(
                f"[Group: {self.group_name}] Failed to generate answer for (seed={q_seed_idx}, instr={instr_idx}, q={question_number})."
            )
            return
        self.file_manager.write_text(answer_file_path, answer_text)
        self.file_manager.write_text(answer_debug_path, final_prompt)
        self.file_manager.write_text(meta_file_path, current_hash)
        Utils.logger.info(f"[Group: {self.group_name}] Saved answer -> {answer_file_path}")

    def process(self):
        if not self.resolve_templates():
            return
        self.collect_instructions_and_seeds()
        file_list = self.group_config.get("files", [])
        if not self.all_question_seeds or not self.all_question_instructions:
            Utils.logger.warning(f"[Group: {self.group_name}] No question seeds or instructions found.")
            return

        # Build file content (could be slightly different for questions/answers)
        combined_content_questions = self.generate_file_content(file_list, for_questions=True)
        combined_content_answers = self.generate_file_content(file_list, for_questions=False)

        # --- Question Generation ---
        question_tasks = []
        question_collections = {}  # (q_seed_idx, instr_idx) -> List[str]
        for q_seed_idx, seed_text in enumerate(self.all_question_seeds, start=1):
            for instr_idx, instruction in enumerate(self.all_question_instructions, start=1):
                question_tasks.append((q_seed_idx, instr_idx, seed_text, instruction))

        def handle_question(task):
            q_seed_idx, instr_idx, seed_text, instruction = task
            text_block = self.generate_question_task(q_seed_idx, instr_idx, seed_text, instruction, combined_content_questions, file_list)
            if not text_block:
                return (q_seed_idx, instr_idx, [])
            parsed = TextParser.parse_questions(text_block)
            return (q_seed_idx, instr_idx, parsed)

        inner_workers = self.thread_count if self.thread_count > 1 else 1
        if inner_workers > 1:
            with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                futures = {pool.submit(handle_question, t): t for t in question_tasks}
                for f in as_completed(futures):
                    q_seed_idx, instr_idx, q_list = f.result()
                    question_collections[(q_seed_idx, instr_idx)] = q_list
        else:
            for t in question_tasks:
                q_seed_idx, instr_idx, q_list = handle_question(t)
                question_collections[(q_seed_idx, instr_idx)] = q_list

        # --- Answer Generation ---
        answer_tasks = []
        for (q_seed_idx, instr_idx), q_list in question_collections.items():
            if not q_list:
                continue
            for q_num, q_text in enumerate(q_list, start=1):
                for answer_instruction in self.all_answer_instructions:
                    answer_tasks.append((q_seed_idx, instr_idx, q_num, q_text, answer_instruction))

        def handle_answer(task):
            q_seed_idx, instr_idx, q_num, q_text, answer_instruction = task
            self.generate_answer(q_seed_idx, instr_idx, q_num, q_text, answer_instruction, combined_content_answers)

        if inner_workers > 1:
            with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                futures = [pool.submit(handle_answer, t) for t in answer_tasks]
                for f in as_completed(futures):
                    f.result()
        else:
            for t in answer_tasks:
                handle_answer(t)
