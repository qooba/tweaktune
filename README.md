<center><img src="./docs/tweaktune_logo_wave_rounded.png" width="100%" alt="Polars logo"></center> 

**tweaktune** is a Rust-powered, Python-facing library for **synthesizing datasets** for **training and fine-tuning AI models**, especially **Language Models**.

Build powerful data pipelines to generate synthetic text, structured JSON, and function calling datasets using LLM APIs. Perfect for creating high-quality training data for model fine-tuning.

[![PyPI](https://img.shields.io/pypi/v/tweaktune.svg)](https://pypi.org/project/tweaktune/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

---

## Documentation

- **[Complete Documentation](docs/README.md)** - Comprehensive guides and API reference
- **[Examples](examples/README.md)** - Working code examples
- **[Getting Started Guide](docs/01-getting-started.md)** - Quick start tutorial

---

## Features

### Flexible Data Sources
Load data from multiple sources:
- **Files**: Parquet, CSV, JSONL, JSON
- **Databases**: PostgreSQL, MySQL, SQLite (via ConnectorX)
- **HuggingFace**: Direct integration with datasets
- **Arrow**: PyArrow datasets and record batches
- **Python**: Dictionaries, functions, Pydantic models
- **APIs**: OpenAPI specifications for function calling
- **SQL**: Filter and transform with SQL queries

### LLM Integration
Connect to any LLM provider:
- **OpenAI**: GPT-4, GPT-3.5, and compatible APIs
- **Azure OpenAI**: Enterprise deployments
- **Local Models**: Unsloth, MistralRS support
- **Custom APIs**: Any OpenAI-compatible endpoint

### Powerful Pipeline Features
- **Parallel Processing**: Multi-worker execution for speed
- **Dynamic Templates**: Jinja2 templating with custom filters
- **Data Validation**: JSON schema, language detection, custom validators
- **Deduplication**: Exact hash, fuzzy simhash, semantic embeddings
- **Quality Checks**: Built-in and custom quality filters
- **Conditional Logic**: If-else branching in pipelines
- **Custom Steps**: Extend with Python classes
- **Metadata Tracking**: Track runs, items, and deduplication state

### Dataset Generation
Create datasets for:
- **Question-Answer pairs**: Synthetic Q&A for training
- **Function calling**: Tool use and API interaction datasets
- **Conversations**: Multi-turn dialogue datasets
- **Structured output**: JSON conforming to schemas
- **Chat formatting**: Model-specific conversation formatting

---

## Quick Start

### Installation

```bash
pip install tweaktune
```

### Simple Example

Generate synthetic data in minutes:

```python
from tweaktune import Pipeline
import os

# Create a Q&A dataset
(Pipeline()
    .with_workers(3)
    .with_llm_openai(
        name="gpt4",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini"
    )
    .with_template("system", "You are an expert educator.")
    .with_template("question", "Generate a question about: {{topic}}")
    .with_template("answer", "Answer this question: {{question}}")
    .with_template("output", """{"topic": {{topic|jstr}}, "question": {{question|jstr}}, "answer": {{answer|jstr}}}""")
    .iter_range(100)
        .add_column("topic", lambda data: f"Topic {data['index']}")
        .generate_text(
            template="question",
            llm="gpt4",
            output="question",
            system_template="system"
        )
        .generate_text(
            template="answer",
            llm="gpt4",
            output="answer",
            system_template="system"
        )
        .write_jsonl(path="qa_dataset.jsonl", template="output")
    .run())
```

### Function Calling Dataset

Create datasets for training models on tool use:

```python
from tweaktune import Pipeline
from pydantic import Field

def search_products(
    query: str = Field(..., description="Search query"),
    category: str = Field(..., description="Product category")
):
    """Search for products in the catalog."""
    pass

(Pipeline()
    .with_workers(5)
    .with_llm_openai("gpt4", api_key, "gpt-4o-mini")
    .with_tools_dataset("tools", [search_products])
    .iter_range(50)
        .sample_tools("tools", 1, "tool")
        # Generate user question, tool call, and response
        # ... (see examples/08_function_calling.py for complete code)
        .render_conversation(
            conversation="@user:question|@assistant:tool_calls([call])|@tool:result|@assistant:answer",
            tools="tool",
            output="conversation"
        )
        .write_jsonl(path="function_calling.jsonl", value="conversation")
    .run())
```

More examples in the [examples](examples/) directory.

---

## Learn More

### Documentation

- [Getting Started](docs/01-getting-started.md) - Installation and first pipeline
- [Pipeline Basics](docs/02-pipeline-basics.md) - Understanding pipelines
- [Data Sources](docs/03-data-sources.md) - Loading data from various sources
- [Templates](docs/04-templates.md) - Using Jinja templates
- [LLM Integration](docs/05-llm-integration.md) - Connecting to LLMs
- [Pipeline Steps](docs/06-pipeline-steps.md) - Complete step reference
- [Custom Steps](docs/07-custom-steps.md) - Creating custom steps
- [Validation & Quality](docs/08-validation-quality.md) - Data validation and deduplication
- [Conversation & Tools](docs/09-conversation-tools.md) - Function calling datasets
- [Chat Templates](docs/10-chat-templates.md) - Formatting for fine-tuning
- [Metadata & Tracking](docs/11-metadata-tracking.md) - Pipeline metadata
- [Advanced Features](docs/12-advanced-features.md) - Advanced patterns

### Examples

- [Simple Pipeline](examples/01_simple_pipeline.py) - Basic usage
- [Data Sources](examples/02_data_sources.py) - Loading various data formats
- [Templates](examples/03_templates.py) - Template examples
- [Transformations](examples/04_transformations.py) - Data transformations
- [Text Generation](examples/05_text_generation.py) - LLM text generation
- [JSON Generation](examples/06_json_generation.py) - Structured output
- [Q&A Dataset](examples/07_qa_dataset.py) - Question-answer pairs
- [Function Calling](examples/08_function_calling.py) - Tool use datasets
- [Conversations](examples/09_conversations.py) - Multi-turn dialogues
- [Deduplication](examples/10_deduplication.py) - Deduplication methods
- [Validation](examples/11_validation.py) - Data validation
- [Custom Steps](examples/12_custom_steps.py) - Custom pipeline steps
- [Conditional Logic](examples/13_conditional_logic.py) - If-else branching
- [Chat Templates](examples/14_chat_templates.py) - Chat formatting

---

## Use Cases

- **Model Fine-tuning**: Generate training datasets for language models
- **Function Calling**: Create datasets for tool use and API interactions
- **Conversation Data**: Build multi-turn dialogue datasets
- **Synthetic Data**: Generate realistic data for testing and development
- **Data Augmentation**: Expand existing datasets with variations
- **Quality Filtering**: Clean and validate datasets with built-in checks
- **RAG Datasets**: Create question-answer pairs from documents

---

## Key Features

### Pipeline Steps

Chain together powerful steps:
- **Data**: `sample()`, `read()`, `filter()`
- **Generation**: `generate_text()`, `generate_json()`, `generate_structured()`
- **Transformation**: `add_column()`, `mutate()`, `map()`, `ifelse()`
- **Validation**: `validate_json()`, `validate_tools()`, `check_language()`
- **Deduplication**: `check_hash()`, `check_simhash()`, `check_embedding()`
- **Rendering**: `render()`, `render_conversation()`, `render_tool_call()`
- **Output**: `write_jsonl()`, `write_csv()`, `print()`
- **Custom**: `.step()` for your own Python classes

### Why tweaktune?

- **Fast**: Rust-powered core for high performance
- **Flexible**: Python API for easy customization
- **Scalable**: Parallel processing with configurable workers
- **Complete**: End-to-end solution from data loading to output
- **Type-safe**: Pydantic integration for structured output
- **Modern**: Built for LLM fine-tuning workflows

---

## Contributing

We welcome contributions! Feel free to open issues, suggest features, or create pull requests.

Please note that by contributing to this project, you agree to the terms of the [Contributor License Agreement (CLA)](CLA.md).

### Development

```bash
# Clone the repository
git clone https://github.com/qooba/tweaktune.git
cd tweaktune

# Build the Python package (development mode)
make pyo3-develop

# Run tests
pytest
```

---

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Links

- **PyPI**: https://pypi.org/project/tweaktune/
- **GitHub**: https://github.com/qooba/tweaktune
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

---

## Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Core performance
- [PyO3](https://pyo3.rs/) - Python bindings
- [Polars](https://pola.rs/) - Fast DataFrames and SQL
- [Minijinja](https://github.com/mitsuhiko/minijinja) - Templating (via minijinja)




