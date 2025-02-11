import sys
# Remove conflicting modules (if any)
modules = list(sys.modules.keys())
for x in modules:
    if "PIL" in x or "google" in x:
        sys.modules.pop(x)

# Patch FastLanguageModel for GRPO training.
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch

# Set model parameters.
max_seq_length = 512  # Increase for longer reasoning traces if needed.
lora_rank = 8         # Larger rank = smarter, but slower.

# Load the base model and tokenizer.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,        # Set to False if you want 16bit LoRA.
    fast_inference=True,      # Enable vLLM fast inference.
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Lower if you run out of memory.
)

# Wrap the model with PEFT (LoRA).
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Suggested values: 8, 16, 32, 64, 128.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long-context fine-tuning.
    random_state=3407,
)

import re
from datasets import load_dataset, Dataset

# ------------------------------------------------------------------------------
# Define a system prompt to instruct the model on the expected answer format.
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# ------------------------------------------------------------------------------
# Helper functions for post-processing model outputs.

def extract_xml_answer(text: str) -> str:
    """Extracts the text within the <answer> ... </answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer if the text uses a hash delimiter."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# ------------------------------------------------------------------------------
# This function assumes that each JSON example has a "messages" field
# which is a list of messages. We take the first user message as the question
# and the first assistant message as the answer.
def get_custom_dataset(split="train") -> Dataset:
    train_data_path = "data.jsonl"
    dataset = load_dataset("json", data_files=train_data_path, split=split)
    
    def format_example(example):
        user_message = None
        assistant_message = None
        # Iterate through the messages and pick the first occurrence of each role.
        for message in example["messages"]:
            if message["role"] == "user" and user_message is None:
                user_message = message["content"]
            elif message["role"] == "assistant" and assistant_message is None:
                assistant_message = message["content"]
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "answer": assistant_message,
        }
    
    # Remove the original columns when mapping
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return dataset

# Load your dataset.
dataset = get_custom_dataset()

# ------------------------------------------------------------------------------
# Define the reward functions to be used by GRPO.

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20,
          f"Question:\n{q}",
          f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# ------------------------------------------------------------------------------
# Configure and start GRPO training.
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm=True,           # Use vLLM for fast inference.
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase if you want smoother training.
    num_generations=6,              # Decrease if you run out of memory.
    max_prompt_length=256,
    max_completion_length=200,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",               # You can integrate with WandB if desired.
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# Start training.
trainer.train()
