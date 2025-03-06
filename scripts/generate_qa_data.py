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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Try importing the new OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("OpenAI package (new interface) not installed; openai provider will not work.")


def get_item_by_name(item_list: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """
    Helper to return the dict that has 'name' == name from a list of dicts.
    """
    for obj in item_list:
        if obj.get("name") == name:
            return obj
    return None


# --- Helper Functions ---
def call_api(
    provider: str,
    model: str,
    prompt: str,
    global_ollama_url: Optional[str] = None,
    openai_client: Optional[OpenAI] = None
) -> Optional[str]:
    """
    Calls the appropriate API (OpenAI or Ollama) using the selected model and prompt.
    Implements an exponential backoff strategy for handling transient failures.
    """
    max_retries = 5
    backoff_factor = 1  # Base backoff time in seconds

    if provider.lower() == "openai":
        if not openai_client:
            logger.error("OpenAI client is not initialized. Cannot generate via openai provider.")
            return None

        attempt = 0
        while attempt <= max_retries:
            try:
                response = openai_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"OpenAI API error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries:
                    return None
                sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying OpenAI API call in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                attempt += 1

    elif provider.lower() == "ollama":
        if not global_ollama_url:
            logger.error("Global Ollama URL is not provided.")
            return None

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {}
        }
        attempt = 0
        while attempt <= max_retries:
            try:
                response = requests.post(global_ollama_url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()

            except Exception as e:
                logger.error(f"Ollama API error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries:
                    return None
                sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying Ollama API call in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                attempt += 1

    else:
        logger.error(f"Unknown provider specified: {provider}")
        return None


def find_file_in_subdirectories(base_dir: Path, relative_path: str) -> Optional[Path]:
    """
    Attempts to locate a file by its relative path in base_dir or its subdirectories.
    """
    possible_path = base_dir / relative_path
    if possible_path.exists():
        return possible_path

    target_name = Path(relative_path).name
    for path in base_dir.rglob(target_name):
        if path.is_file():
            return path
    return None


def get_hash(text: str) -> str:
    """Returns the SHA-256 hash of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text_to_file(file_path: Path, text: str) -> None:
    """Writes text to a file ensuring the parent directory exists."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def read_text_from_file(file_path: Path) -> str:
    """Reads and returns text from a file."""
    return file_path.read_text(encoding="utf-8")


def build_files_content(
    file_list: List[str],
    base_dir: Path,
    file_header_template: str
) -> str:
    """
    Build a combined string by iterating over a list of files, 
    inserting a file header and then the file contents.
    """
    combined = ""
    for rel_path in file_list:
        file_path = find_file_in_subdirectories(base_dir, rel_path)
        if file_path and file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            combined += file_header_template.format(file_name=rel_path) + "\n"
            combined += content + "\n\n"
        else:
            logger.warning(f"{rel_path} not found in {base_dir} or its subdirectories.")
    return combined


def process_file_group(
    group_name: str,
    group_config: Dict[str, Any],
    config: Dict[str, Any],
    full_base_dir: Path,
    base_output_path: Path,
    question_provider_config: Dict[str, Any],
    answer_provider_config: Dict[str, Any],
    global_ollama_url: Optional[str],
    openai_client: Optional[OpenAI],
    global_thread_count: int
) -> None:
    """
    Processes a group of files to generate questions and answers using the new config format.
    """

    # 1) Gather the relevant config sections/lists
    file_headers = config.get("FileHeaders", [])
    question_prompts = config.get("QuestionPrompt", [])
    answer_prompts = config.get("AnswerPrompt", [])
    question_instruction_lists = config.get("QuestionInstructionList", [])
    answer_instruction_lists = config.get("AnswerInstructionList", [])
    generate_question_lists = config.get("GenerateQuestionLists", [])

    # 2) Group-level info
    file_list: List[str] = group_config.get("files", [])
    file_header_name = group_config.get("file_header", "")
    question_prompt_name = group_config.get("question_prompt", "")
    answer_prompt_name = group_config.get("answer_prompt", "")

    # 2a) Instruction list names
    question_instruction_list_names = group_config.get("question_instruction_list", [])
    answer_instruction_list_names = group_config.get("answer_instruction_list", [])

    # 2b) Generate question list names
    generate_question_list_names = group_config.get("generate_question_list", [])

    # 3) Resolve the actual template strings
    file_header_obj = get_item_by_name(file_headers, file_header_name)
    file_header_template = file_header_obj["description"] if file_header_obj else ""

    question_prompt_obj = get_item_by_name(question_prompts, question_prompt_name)
    if not question_prompt_obj:
        logger.error(f"No question prompt found for name '{question_prompt_name}'.")
        return
    question_prompt_template = question_prompt_obj["description"]

    answer_prompt_obj = get_item_by_name(answer_prompts, answer_prompt_name)
    if not answer_prompt_obj:
        logger.error(f"No answer prompt found for name '{answer_prompt_name}'.")
        return
    answer_prompt_template = answer_prompt_obj["description"]

    # 4) Collect all instructions from the relevant lists
    #    We'll generate questions for each combination of question-lists x question-instructions
    #    and then generate answers for each question x answer-instructions
    all_question_instructions = []
    for q_list_name in question_instruction_list_names:
        instr_list_obj = get_item_by_name(question_instruction_lists, q_list_name)
        if instr_list_obj:
            all_question_instructions.extend(instr_list_obj.get("instruction", []))

    all_answer_instructions = []
    for a_list_name in answer_instruction_list_names:
        instr_list_obj = get_item_by_name(answer_instruction_lists, a_list_name)
        if instr_list_obj:
            all_answer_instructions.extend(instr_list_obj.get("instruction", []))

    # 5) Collect all "question seeds" from the relevant generate question lists
    all_question_seeds = []
    for gq_list_name in generate_question_list_names:
        gq_obj = get_item_by_name(generate_question_lists, gq_list_name)
        if gq_obj:
            all_question_seeds.extend(gq_obj.get("questions", []))

    # If no seeds or instructions, we won't do anything
    if not all_question_seeds or not all_question_instructions:
        logger.warning(f"[Group: {group_name}] No question seeds or instructions found.")
        return

    # 6) Build file content for the prompt (shared for question generation)
    combined_file_content_for_questions = build_files_content(file_list, full_base_dir, file_header_template)

    # 7) Prepare output directories
    output_dir = base_output_path / "qa_generation_output"
    questions_dir = output_dir / "questions"
    answers_dir = output_dir / "answers"
    debug_dir = output_dir / "debug"
    questions_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 8) We'll generate a single question for each (question_seed, question_instruction)
    def generate_questions(q_seed_idx: int, instr_idx: int, seed_text: str, instruction: str):
        """
        Generate a single question from a single question seed + instruction pair.
        Return the text as the question.
        """
        # Format the final question prompt
        file_name_list_str = ", ".join(file_list)
        final_question_prompt = question_prompt_template.format(
            file_content=combined_file_content_for_questions,
            generate_question=seed_text,
            instruction=instruction,
            file_name_list=file_name_list_str
        )

        out_filename = f"questions_{group_name}_seed{q_seed_idx}_instr{instr_idx}.txt"
        debug_filename = f"debug_{group_name}_seed{q_seed_idx}_instr{instr_idx}_questions.txt"
        questions_path = questions_dir / out_filename
        debug_path = debug_dir / debug_filename

        if questions_path.exists():
            # If we already have this questions file, reuse it
            question_text = read_text_from_file(questions_path).strip()
            logger.info(f"[Group: {group_name}] Using existing questions file: {out_filename}")
        else:
            # Call the LLM
            question_text = call_api(
                provider=question_provider_config.get("provider"),
                model=question_provider_config.get("model"),
                prompt=final_question_prompt,
                global_ollama_url=global_ollama_url,
                openai_client=openai_client
            )
            if not question_text:
                logger.error(f"[Group: {group_name}] Failed to generate question (seed={q_seed_idx}, instr={instr_idx}).")
                return ""
            # Save outputs
            write_text_to_file(questions_path, question_text)
            write_text_to_file(debug_path, final_question_prompt)

        return question_text

    # Collect tasks for concurrency
    question_generation_tasks = []
    for q_seed_idx, seed_text in enumerate(all_question_seeds, start=1):
        for instr_idx, instruction in enumerate(all_question_instructions, start=1):
            question_generation_tasks.append((q_seed_idx, instr_idx, seed_text, instruction))

    # Decide how many workers to use
    inner_workers = global_thread_count if global_thread_count > 1 else 1

    # We'll store (seed_idx, instr_idx) -> single question text (wrapped in a list)
    question_collections: Dict[(int, int), List[str]] = {}

    def handle_question_generation_task(task_tuple):
        q_seed_idx, instr_idx, seed_text, instruction = task_tuple
        text_block = generate_questions(q_seed_idx, instr_idx, seed_text, instruction)
        if not text_block:
            return (q_seed_idx, instr_idx, [])
        # Use the entire output as the question, wrapped in a list
        return (q_seed_idx, instr_idx, [text_block])

    # Generate questions concurrently or sequentially
    if inner_workers > 1:
        with ThreadPoolExecutor(max_workers=inner_workers) as pool:
            future_map = {pool.submit(handle_question_generation_task, t): t for t in question_generation_tasks}
            for f in as_completed(future_map):
                (q_seed_idx, instr_idx, q_list) = f.result()
                question_collections[(q_seed_idx, instr_idx)] = q_list
    else:
        for t in question_generation_tasks:
            q_seed_idx, instr_idx, q_list = handle_question_generation_task(t)
            question_collections[(q_seed_idx, instr_idx)] = q_list

    # --- ANSWER GENERATION ---
    # Build file content again for answers
    combined_file_content_for_answers = build_files_content(file_list, full_base_dir, file_header_template)

    def generate_answer(
        q_seed_idx: int,
        instr_idx: int,
        question_number: int,
        question_text: str,
        answer_instruction: str
    ):
        """
        Generate an answer to a single question with a single answer instruction.
        """
        final_answer_prompt = answer_prompt_template.format(
            file_content=combined_file_content_for_answers,
            instruction=answer_instruction,
            question=question_text
        )
        
        ans_instr_hash = get_hash(answer_instruction)[:8]
        answer_filename = f"answer_{group_name}_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{ans_instr_hash}.txt"
        debug_filename = f"debug_{group_name}_answer_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{ans_instr_hash}.txt"
        meta_filename = f"answer_{group_name}_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{ans_instr_hash}.meta"

        answer_file_path = answers_dir / answer_filename
        answer_debug_path = debug_dir / debug_filename
        meta_file_path = answers_dir / meta_filename

        current_hash = get_hash(final_answer_prompt)

        regenerate = True
        if answer_file_path.exists():
            if meta_file_path.exists():
                stored_hash = read_text_from_file(meta_file_path).strip()
                if stored_hash == current_hash:
                    logger.info(
                        f"[Group: {group_name}] Answer for (seed={q_seed_idx}, instr={instr_idx}, q={question_number}) is up to date."
                    )
                    regenerate = False
                else:
                    logger.info(f"[Group: {group_name}] Detected changed question/instruction, regenerating answer.")
            else:
                write_text_to_file(meta_file_path, current_hash)
                regenerate = False

        if not regenerate:
            return

        answer_text = call_api(
            provider=answer_provider_config.get("provider"),
            model=answer_provider_config.get("model"),
            prompt=final_answer_prompt,
            global_ollama_url=global_ollama_url,
            openai_client=openai_client
        )
        if not answer_text:
            logger.error(
                f"[Group: {group_name}] Failed to generate answer for (seed={q_seed_idx}, instr={instr_idx}, q={question_number})."
            )
            return

        write_text_to_file(answer_file_path, answer_text)
        write_text_to_file(answer_debug_path, final_answer_prompt)
        write_text_to_file(meta_file_path, current_hash)
        logger.info(f"[Group: {group_name}] Saved answer -> {answer_file_path}")

    answer_tasks = []
    for (q_seed_idx, instr_idx), question_list in question_collections.items():
        if not question_list:
            continue
        for q_num, q_text in enumerate(question_list, start=1):
            for ans_instr in all_answer_instructions:
                answer_tasks.append((q_seed_idx, instr_idx, q_num, q_text, ans_instr))

    
    def handle_answer_task(task_tuple):
        q_seed_idx, instr_idx, question_number, question_text, answer_instruction = task_tuple
        generate_answer(q_seed_idx, instr_idx, question_number, question_text, answer_instruction)

    if inner_workers > 1:
        with ThreadPoolExecutor(max_workers=inner_workers) as pool:
            _ = list(as_completed(pool.submit(handle_answer_task, t) for t in answer_tasks))
    else:
        for t in answer_tasks:
            handle_answer_task(t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA data for LLM fine-tuning.")
    parser.add_argument("--config", default="generate_qa_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--threads", type=int, default=8, help="Max workers for processing all tasks")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    global_config = config.get("global", {})
    base_dir = global_config.get("base_dir", "")
    output_base_path = Path(global_config.get("output_base_path", "/var/kolo_data"))
    full_base_dir = output_base_path / base_dir
    global_ollama_url = global_config.get("ollama_url", "http://localhost:11434/api/generate")

    question_provider_config = config.get("providers", {}).get("question", {})
    answer_provider_config = config.get("providers", {}).get("answer", {})

    openai_client = None
    if (
        question_provider_config.get("provider", "").lower() == "openai"
        or answer_provider_config.get("provider", "").lower() == "openai"
    ):
        if OpenAI is None:
            logger.error("OpenAI client cannot be initialized because the package is missing.")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            openai_client = OpenAI(api_key=api_key)

    file_groups_config = config.get("file_groups", {})

    expanded_file_groups = {}
    for group_name, g_config in file_groups_config.items():
        iterations = g_config.get("iterations", 1)
        for i in range(1, iterations + 1):
            key = f"{group_name}_{i}"
            expanded_file_groups[key] = g_config

    total_groups = len(expanded_file_groups)
    logger.info(f"Starting processing of {total_groups} file groups with up to {args.threads} threads...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for group_name, group_conf in expanded_file_groups.items():
            future = executor.submit(
                process_file_group,
                group_name=group_name,
                group_config=group_conf,
                config=config,
                full_base_dir=full_base_dir,
                base_output_path=output_base_path,
                question_provider_config=question_provider_config,
                answer_provider_config=answer_provider_config,
                global_ollama_url=global_ollama_url,
                openai_client=openai_client,
                global_thread_count=args.threads
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()

    logger.info("All file groups have been processed successfully.")


if __name__ == "__main__":
    main()
