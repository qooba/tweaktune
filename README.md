# tweaktune

**tweaktune** is a Rust-powered, Python-facing library designed to **synthesize datasets** for **training and fine-tuning AI models**, especially **LMs** (Language Models).  
It allows you to easily build data pipelines, generate new examples using LLM APIs, and create structured datasets from a variety of sources.

---

## Features

- **Flexible Data Sources**:  
  Supports datasets from:
  - Parquet files
  - CSV files
  - JSONL files
  - Arrow datasets
  - OpenAPI specifications (for function calling datasets)
  - Lists of tools (Python functions for function calling datasets)
  - Pydantic models (for structured output datasets)

- **LLM Integration**:  
  Connects to any LLM API to generate synthetic text or structured JSON.

- **Dynamic Prompting**:  
  Supports **Jinja templates** for highly customizable prompts.

- **Parallel Processing**:  
  Configure **multiple workers** to run your pipeline steps in parallel.

- **Easy Pipeline Building**:  
  Compose steps like sampling, generating, writing, or debugging into a seamless pipeline.

---

## Quick Example

Here's how you can build a dataset from a Parquet file and synthesize new data using an LLM API:

```python
from tweaktune import Pipeline
import os

persona_template = """
Na podstawie poniższego fragmentu tekstu opisz personę która jest z nim związana.
Dla opisywanej osoby wymyśl fikcyjne imię i nazwisko.
Napisz dwa zdania na temat tej osoby, opis zwróć w formacie json, nie dodawaj nic więcej:
{"persona":"opis osoby"}

---
FRAGMENT TEKSTU:

{{article[0].text}}
"""

url = "http://localhost:8000/"
api_key = os.environ["API_KEY"]
model = "model"

p = Pipeline()\
    .with_workers(5)\
    .with_parquet_dataset("web_articles", "../../datasets/articles.pq")\
    .with_llm_api("bielik", url, api_key, model)\
    .with_template("persona", persona_template)\
    .with_template("output", """{"persona": {{persona|jstr}} }""")\
    .iter_range(10000)\
        .sample(dataset="web_articles", size=1, output="article")\
        .generate_json(template="persona", llm="bielik", output="persona", json_path="persona")\
        .write_jsonl(path="../../datasets/personas.jsonl", template="output")\
    .run()
```

---

## Pipeline Steps

You can easily chain together multiple steps:

- `sample()` – sample items from a dataset
- `read()` – read entire dataset
- `generate_text()` – generate text using an LLM
- `generate_json()` – generate JSON output and extract a specific field
- `write_jsonl()` – write output to a JSONL file
- `write_csv()` – write output to a CSV file
- `print()` – print outputs
- `debug()` – enable detailed debugging
- `log()` – set log level
- `python step` – add custom Python-defined step classes

---

## Why tweaktune?

- Build synthetic datasets faster for fine-tuning models.
- Automate text, JSON, or structured data generation.
- Stay flexible: plug your own LLM API or use existing OpenAI-compatible ones.
- Rust speed, Python usability.

---

## 📦 Installation

```bash
pip install tweaktune
```


## 🤝 Contributing

We welcome contributions! Feel free to open issues, suggest features, or create pull requests.

Please note that by contributing to this project, you agree to the terms of the [Contributor License Agreement (CLA)](CLA.md).




