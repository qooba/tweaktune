# tweaktune Examples

This directory contains practical examples demonstrating various tweaktune features.

## Basic Examples

- [01_simple_pipeline.py](01_simple_pipeline.py) - Simple data generation pipeline
- [02_data_sources.py](02_data_sources.py) - Loading data from various sources
- [03_templates.py](03_templates.py) - Using Jinja templates
- [04_transformations.py](04_transformations.py) - Data transformation steps

## LLM Examples

- [05_text_generation.py](05_text_generation.py) - Generate text with LLMs
- [06_json_generation.py](06_json_generation.py) - Generate structured JSON
- [07_qa_dataset.py](07_qa_dataset.py) - Create question-answer datasets

## Advanced Examples

- [08_function_calling.py](08_function_calling.py) - Generate function calling datasets
- [09_conversations.py](09_conversations.py) - Build conversation datasets
- [10_deduplication.py](10_deduplication.py) - Data deduplication with metadata
- [11_validation.py](11_validation.py) - Data validation and quality checks
- [12_custom_steps.py](12_custom_steps.py) - Create custom pipeline steps
- [13_conditional_logic.py](13_conditional_logic.py) - Conditional branching with ifelse

## Training Examples

- [14_chat_templates.py](14_chat_templates.py) - Format data for model fine-tuning
- [15_rl_formats.py](15_rl_formats.py) - Generate datasets for RL methods (SFT, DPO, GRPO)

## Running Examples

Each example is self-contained and can be run independently:

```bash
python examples/01_simple_pipeline.py
```

Some examples require environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
python examples/05_text_generation.py
```

## Requirements

Install tweaktune and optional dependencies:

```bash
pip install tweaktune

# For LLM examples
pip install openai

# For HuggingFace examples
pip install datasets transformers

# For database examples
pip install connectorx

# For UI mode
pip install nicegui
```

## Output

Examples write output to the current directory. Clean up with:

```bash
rm -f *.jsonl *.csv
```
