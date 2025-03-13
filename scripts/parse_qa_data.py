import os
import json
import glob
import argparse
import yaml

from SyntheticDataGeneration.Utils import Utils  # Import the Utils class with the logger

# OUTPUT_FILE path remains unchanged.
OUTPUT_FILE = "/app/data.jsonl"

def pair_questions_and_answers(config, answers_dir):
    """
    Iterates over file groups defined in the config, expands each group based on its iterations,
    and pairs each question (from the YAML config's question_list) with its corresponding answer files.
    
    Expected answer file naming:
      answer_{expanded_group}_q{question_number}_{hash}.txt
    where expanded_group is of the form {group_name}_{iteration}.
    """
    qa_pairs = []
    group_stats = {}  # { identifier: {'questions': count, 'answers': count} }

    file_groups = config.get("file_groups", {})
    for group_name, group_conf in file_groups.items():
        iterations = group_conf.get("iterations", 1)
        question_list = group_conf.get("question_list", [])
        if not question_list:
            Utils.logger.warning(f"No question_list found for file group '{group_name}'. Skipping.")
            continue

        for i in range(1, iterations + 1):
            expanded_group = f"{group_name}_{i}"
            identifier = expanded_group
            group_stats[identifier] = {'questions': len(question_list), 'answers': 0}

            for idx, question in enumerate(question_list, start=1):
                # Expected answer file format: answer_{expanded_group}_q{idx}_*.txt
                pattern = os.path.join(
                    answers_dir,
                    f"answer_{expanded_group}_q{idx}_*.txt"
                )
                matching_files = glob.glob(pattern)
                if not matching_files:
                    Utils.logger.warning(f"No answer file found for identifier {identifier}, question {idx}.")
                    continue

                for answer_filepath in matching_files:
                    Utils.logger.info(f"Processing answer file: {answer_filepath} for question {idx} in group: {identifier}")
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
    parser = argparse.ArgumentParser(
        description="Pair answer files with questions from the YAML config and generate a JSONL training data output."
    )
    parser.add_argument(
        "--qa_output", 
        type=str, 
        default="qa_generation_output",
        help="Base directory for QA generation output. Default is 'qa_generation_output'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="generate_qa_config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    config_file = args.config
    if not os.path.exists(config_file):
        Utils.logger.error(f"Configuration file {config_file} does not exist.")
        return

    with open(config_file, 'r', encoding='utf-8') as cf:
        config = yaml.safe_load(cf)

    # Set directory for answer files based on the provided qa_output base directory.
    base_output_dir = args.qa_output
    answers_dir = os.path.join(base_output_dir, "answers")

    qa_pairs, group_stats = pair_questions_and_answers(config, answers_dir)
    
    if not qa_pairs:
        Utils.logger.info("No QA pairs found.")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for pair in qa_pairs:
            json_line = json.dumps(pair, ensure_ascii=False)
            out_f.write(json_line + "\n")

    # Log summary statistics.
    total_questions = 0
    total_answers = 0
    Utils.logger.info("Processing Summary:")
    for identifier, stats in group_stats.items():
        total_questions += stats['questions']
        total_answers += stats['answers']
        Utils.logger.info(f"  Identifier '{identifier}': {stats['questions']} questions, {stats['answers']} answers processed.")

    Utils.logger.info(f"Total: {total_questions} questions and {total_answers} answers processed.")
    Utils.logger.info(f"Total QA pairs saved to {OUTPUT_FILE}: {len(qa_pairs)}")

if __name__ == "__main__":
    main()
