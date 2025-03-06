import os
import json
import re
import glob
from typing import List

# Adjust these paths as needed.
BASE_OUTPUT_DIR = "/var/kolo_data/qa_generation_output"
QUESTIONS_DIR = os.path.join(BASE_OUTPUT_DIR, "questions")
ANSWERS_DIR = os.path.join(BASE_OUTPUT_DIR, "answers")
OUTPUT_FILE = "/app/data.jsonl"

def parse_question_from_file(filepath: str) -> List[str]:
    """
    Reads the entire file content and returns it as a single question.
    Since each file now contains a complete question, no further parsing is needed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return [content] if content else []

def pair_questions_and_answers():
    """
    Looks for all question files in the QUESTIONS_DIR and then for each question,
    pairs it with the corresponding answer files from ANSWERS_DIR.

    Assumes the naming convention:
      - Questions: questions_{group_name}_seed{q_seed_idx}_instr{instr_idx}.txt
      - Answers:   answer_{group_name}_seed{q_seed_idx}_instr{instr_idx}_q{question_number}_{hash}.txt

    If there are multiple answer files for a given question, each answer is saved as its own QA pair.
    """
    qa_pairs = []
    group_stats = {}  # { identifier: {'questions': count, 'answers': count} }

    # Process each question file.
    for q_filename in os.listdir(QUESTIONS_DIR):
        # Expect filenames like: questions_{group_name}_seed{q_seed_idx}_instr{instr_idx}.txt
        m = re.match(r"questions_(.+)_seed(\d+)_instr(\d+)\.txt", q_filename)
        if not m:
            continue

        group_name = m.group(1)
        q_seed_idx = m.group(2)
        instr_idx = m.group(3)
        identifier = f"{group_name}_seed{q_seed_idx}_instr{instr_idx}"
        q_filepath = os.path.join(QUESTIONS_DIR, q_filename)
        questions = parse_question_from_file(q_filepath)

        # Initialize stats for this identifier.
        group_stats[identifier] = {'questions': len(questions), 'answers': 0}

        # For each question, look for the corresponding answer files using a glob pattern.
        # Expected answer file format: answer_{group_name}_seed{q_seed_idx}_instr{instr_idx}_q{idx}_{hash}.txt
        for idx, question in enumerate(questions, start=1):
            pattern = os.path.join(
                ANSWERS_DIR,
                f"answer_{group_name}_seed{q_seed_idx}_instr{instr_idx}_q{idx}_*.txt"
            )
            matching_files = glob.glob(pattern)
            if not matching_files:
                print(f"Warning: No answer file found for identifier {identifier}, question {idx}.")
                continue

            for answer_filepath in matching_files:
                with open(answer_filepath, 'r', encoding='utf-8') as af:
                    answer = af.read().strip()
                # Each answer file gets its own Q&A pair.
                qa_pair = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
                qa_pairs.append(qa_pair)
                group_stats[identifier]['answers'] += 1

    return qa_pairs, group_stats

def main():
    qa_pairs, group_stats = pair_questions_and_answers()

    if not qa_pairs:
        print("No QA pairs found.")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for pair in qa_pairs:
            json_line = json.dumps(pair, ensure_ascii=False)
            out_f.write(json_line + "\n")

    # Print summary statistics.
    total_questions = 0
    total_answers = 0
    print("Processing Summary:")
    for identifier, stats in group_stats.items():
        total_questions += stats['questions']
        total_answers += stats['answers']
        print(f"  Identifier '{identifier}': {stats['questions']} questions, {stats['answers']} answers processed.")

    print(f"Total: {total_questions} questions and {total_answers} answers processed.")
    print(f"Total QA pairs saved to {OUTPUT_FILE}: {len(qa_pairs)}")

if __name__ == "__main__":
    main()
