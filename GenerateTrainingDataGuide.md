# Generate Synthetic Training Data Guide

## Overview

The **Kolo** project uses the following scripts and configuration file to generate and process QA data:

1. The following command will copy over all subfolders, documents and files into `/var/kolo_data/qa_generation_input`.

   ```bash
   ./copy_qa_input_generation.ps1 "directory"
   ```

   If you are testing for the first time. Try copying the entire Kolo project by running this command.

   ```bash
   ./copy_qa_input_generation.ps1 "../"
   ```

1. Modify the [config file](https://github.com/MaxHastings/Kolo/blob/main/scripts/generate_qa_config.yaml) to specify file groups, custom prompts, and the number of iterations. If you are testing with Kolo project, leave the config file untouched.

1. Run the copy all scripts command. This will move the configuration file into Kolo.

   ```bash
   ./copy_scripts.ps1
   ```

1. This will generate QA data using the LLM provider and model you choose. In the config file you can choose whether to use `openai` or `ollama` and the specified model name. By default we use `openai` and the model `gpt-4o-mini`. When using the OpenAI provider you must pass in your API key when running the generating script.

   ```bash
   ./generate_qa_data.ps1 -OPENAI_API_KEY "your key"
   ```

   Multi-threaded parameters

   ```bash
   ./generate_qa_data.ps1 -OPENAI_API_KEY "your key" -Threads 16
   ```

1. After generating the QA prompts, this command converts the question and answer text files inside  
   `/var/kolo_data/qa_generation_output` into training data: `data.jsonl` and `data.json` in `/app/`.

   ```bash
   ./convert_qa_output.ps1
   ```

   Note: On subsequent generations, ensure you delete the existing `qa_generation_output` folder by executing:

   ```bash
   ./delete_qa_generation_output.ps1
   ```

1. Your training data is now ready; continue by training your LLM using `./train_model_torchtune.ps1` or `./train_model_unsloth.ps1`.  
   Follow the README guide after this step.

---

# Config File Details

This YAML configuration file controls various aspects of the QA generation process.

## Global Settings

### Directories & Paths

- **`base_dir`**: Location of the QA generation input files.
- **`output_dir`**: Directory where QA generation output and debug files are saved.
- **`output_base_path`**: The base path for output files (e.g., `/var/kolo_data`).

### Service Endpoints

- **`ollama_url`**: URL endpoint for the Ollama API (if used).

## Providers

Define the API providers for generating both questions and answers. Each provider block specifies:

- **`provider`**: The service to use (e.g., `openai` or `ollama`).
- **`model`**: The model to be used (e.g., `gpt-4o-mini`).

```
global:
  base_dir: qa_generation_input
  output_dir: qa_generation_output
  output_base_path: /var/kolo_data
  ollama_url: http://localhost:11434/api/generate

providers:
  question:
    provider: openai # Use "ollama" or "openai"
    model: gpt-4o-mini
  answer:
    provider: openai # Use "ollama" or "openai"
    model: gpt-4o-mini
```

## Prompts

### Instruction Lists

#### Question Instruction List

This list defines different instructions to style the generated questions. Each entry may have multiple instructions. For example:

```
QuestionInstructionList:
  - name: 'ToneInstructions'
    instruction:
      - 'Keep it short.'
      - ''
      - 'Use slang.'
      - 'Use broken english.'
      - 'Sound like a developer.'
```

Usage: During question generation, each instruction is applied to a seed to create variations in questions, in the above example using different tones.

#### Answer Instruction List

This list provides variations in the answer generation style:

```
AnswerInstructionList:
  - name: 'SimpleAndComplex'
    instruction:
      - 'For your answer keep it simple and short.'
      - 'For your answer give detail and reference any relevant content.'
```

Usage: Each answer instruction is paired with a question to generate answers with different levels of complexity or detail.

### Question Generation Seeds

The GenerateQuestionLists section provides seed questions or prompts that drive the question generation process:

```
GenerateQuestionLists:
  - name: 'DocumentList'
    questions:
      - 'Write a question as if you are an experienced user with made up specific requirements and details and you want the LLM to work for you using the following information.'
      - 'Write a question as if you are a new user and you want to know how to do something from the following information.'
      - 'Write a question as if you are a new user and you want to know something specific from the following information.'
      - 'Write a question as if you are a new user and you want to have a broad understanding of the following information.'
      - 'Write a question as if you are a new user that is confused and wants something rephrased from the following information.'

  - name: 'CodingList'
    questions:
      - 'Write a question as if you are an experienced user that wants to know a specific implementation detail in the following code.'
      - 'Write a question as if you are a new user that wants to know how to use the following code for a particular situation.'
```

Usage: The seeds are combined with the instructions to produce a diverse set of questions, such as in the above example tailoring them to either documentation or coding contexts.

### Prompt Templates

Prompt templates are used to construct the text sent to the language model.

#### FileHeaders

The FileHeaders section specifies the header prompt that will be inserted above each file content.

```
FileHeaders:
  - name: 'DefaultFileHeader'
    description: 'The file contents for: {file_name}'
```

- {file_name}: Represents the file name.

#### Answer Prompt

Defines how to format the answer prompt:

```
AnswerPrompt:
  - name: 'DefaultAnswerPrompt'
    description: |
      {question}
      {instruction}
      ---
      {file_content}
```

Usage: Placeholders are replaced as follows:

- {file_content}: Combined content of the source files.
- {instruction}: The answer instruction text.
- {question}: The specific question to answer.

#### QuestionPrompt

Defines how to format the question prompt:

For this example, there are two variants for question prompts, depending on whether the file names should be referenced or not.

```
QuestionPrompt:
  - name: 'NoFileName'
    description: |
      {generate_question}
      {instruction}
      ---
      {file_content}
  - name: 'WithFileName'
    description: |
      {generate_question}
      {instruction}
      You are required to reference {file_name_list} in the question that you create!
      ---
      {file_content}
```

Usage:

- {file_content}: Combined content of the source files.
- {instruction}: The question instruction text.
- {generate_question}: The specific generate question instruction from the Generate Question List.
- {file_name_list} is the list of file names that you can use to instruct the LLM to use when generating questions.

Note: Changing the output format may impact how well the conversion script works.

### File Groups

The file_groups section organizes the files into groups that will each be processed independently. Each file group defines:

- iterations: How many times the group should be processed (each iteration may generate a new set of Q&A outputs).
- files: List of files that you want to use for the LLM context.
- question_prompt: Which question prompt template to use (e.g., NoFileName or WithFileName).
- generate_question_list: Which question generation seed list(s) to use.
- question_instruction_list: Which instruction list to apply when generating a question.
- file_header: Which file header template to use.
- answer_prompt: Which answer prompt template to use.
- answer_instruction_list: Which answer instruction list to apply when generating an answer.

Example configuration for three groups:

```
file_groups:
  UninstallModel:
    iterations: 10
    files:
      - uninstall_model.ps1
    question_prompt: WithFileName
    generate_question_list: [CodingList]
    question_instruction_list: [ToneInstructions]
    file_header: DefaultFileHeader
    answer_prompt: DefaultAnswerPrompt
    answer_instruction_list: [Default]
  README:
    iterations: 30
    files:
      - README.md
    question_prompt: NoFileName
    generate_question_list: [DocumentList]
    question_instruction_list: [ToneInstructions]
    file_header: DefaultFileHeader
    answer_prompt: DefaultAnswerPrompt
    answer_instruction_list: [Default]
```

See [generate_qa_config.yaml](https://github.com/MaxHastings/Kolo/blob/main/scripts/generate_qa_config.yaml) for a full config example.

## Debugging

If you run into issues, you can look at the debug folder inside `kolo_container` at `/var/kolo_data/qa_generation_output` using WinSCP. The debug text files will show you exactly what is being sent to the LLM during generation.
