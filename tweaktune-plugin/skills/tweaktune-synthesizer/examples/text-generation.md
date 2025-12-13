# Text Generation Examples

This document provides examples of text generation pipelines using tweaktune.

## Basic Text Generation

Generate articles from topics with deduplication:

```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/articles.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build and run pipeline
    (Pipeline(name="text-generation", metadata=Metadata(description="Generate articles"))
        .with_workers(4)
        .with_jsonl_dataset("topics", "topics.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("prompt", "Generate a detailed article about: {{topic}}")
        .iter_dataset("topics")
        .generate_text(
            template="prompt",
            llm="gpt4",
            output="article",
            max_tokens=2048,
            temperature=0.7
        )
        .check_hash("article")  # Deduplication
        .write_jsonl(
            path=str(output_path),
            template='{"topic": "{{topic}}", "article": "{{article}}"}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

**Input (topics.jsonl):**
```json
{"topic": "Machine Learning"}
{"topic": "Quantum Computing"}
{"topic": "Climate Change"}
```

**Output (output/articles.jsonl):**
```json
{"topic": "Machine Learning", "article": "Machine learning is..."}
{"topic": "Quantum Computing", "article": "Quantum computing represents..."}
{"topic": "Climate Change", "article": "Climate change is..."}
```

## Multi-Field Generation

Generate multiple fields per example (title, summary, body):

```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/articles_multifield.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="multi-field-generation", metadata=Metadata(description="Generate multi-field articles"))
        .with_workers(4)
        .with_jsonl_dataset("topics", "topics.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        # Define templates for each field
        .with_template("title_prompt", "Generate a catchy title for an article about: {{topic}}")
        .with_template("summary_prompt", "Write a 2-sentence summary for an article titled '{{title}}' about {{topic}}")
        .with_template("body_prompt", "Write a detailed article with the title '{{title}}' and summary '{{summary}}' about {{topic}}")
        .iter_dataset("topics")
        # Generate title
        .generate_text(
            template="title_prompt",
            llm="gpt4",
            output="title",
            max_tokens=50,
            temperature=0.8
        )
        # Generate summary based on title
        .generate_text(
            template="summary_prompt",
            llm="gpt4",
            output="summary",
            max_tokens=100,
            temperature=0.7
        )
        # Generate full body based on title and summary
        .generate_text(
            template="body_prompt",
            llm="gpt4",
            output="body",
            max_tokens=1500,
            temperature=0.7
        )
        .check_hash("body")  # Deduplication
        .write_jsonl(
            path=str(output_path),
            template='{"topic": "{{topic}}", "title": "{{title}}", "summary": "{{summary}}", "body": "{{body}}"}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## With Quality Checks

Add deduplication and language filtering:

```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/articles_quality.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="text-gen-with-quality", metadata=Metadata(description="Generate with quality checks"))
        .with_workers(4)
        .with_jsonl_dataset("topics", "topics.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("prompt", "Generate a detailed article in English about: {{topic}}")
        .iter_dataset("topics")
        .generate_text(
            template="prompt",
            llm="gpt4",
            output="article",
            max_tokens=2048,
            temperature=0.7
        )
        # Quality checks
        .check_hash("article")  # Exact deduplication
        .check_simhash("article", threshold=0.95)  # Fuzzy deduplication
        .check_language(
            input="article",
            language="english",
            precision=0.9
        )  # Language filtering
        .write_jsonl(
            path=str(output_path),
            template='{"topic": "{{topic}}", "article": "{{article}}"}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## Using Jinja2 Templates

For complex prompts, use external Jinja2 template files:

**templates/article_prompt.j2:**
```jinja
You are an expert technical writer.

Write a comprehensive article about {{topic}}.

The article should:
- Be at least 500 words
- Include an introduction, body, and conclusion
- Use clear, engaging language
- Include examples where appropriate

Topic: {{topic}}
{% if context %}
Context: {{context}}
{% endif %}

Article:
```

**pipeline.py:**
```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/articles_jinja.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="text-gen-jinja", metadata=Metadata(description="Generate with Jinja2 templates"))
        .with_workers(4)
        .with_jsonl_dataset("topics", "topics.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_j2_template("article_prompt", "templates/article_prompt.j2")
        .iter_dataset("topics")
        .generate_text(
            template="article_prompt",
            llm="gpt4",
            output="article",
            max_tokens=2048,
            temperature=0.7
        )
        .check_hash("article")
        .write_jsonl(
            path=str(output_path),
            template='{"topic": "{{topic}}", "article": "{{article}}"}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## From Scratch (No Input Data)

Generate text without seed data using `.iter_range()`:

```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/stories.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="stories-from-scratch", metadata=Metadata(description="Generate stories from scratch"))
        .with_workers(4)
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("story_prompt", "Generate a unique short story. Be creative and diverse.")
        .iter_range(100)  # Generate 100 stories
        .generate_text(
            template="story_prompt",
            llm="gpt4",
            output="story",
            max_tokens=1000,
            temperature=0.9  # Higher temperature for creativity
        )
        .check_simhash("story", threshold=0.90)  # Fuzzy dedup for diverse stories
        .write_jsonl(
            path=str(output_path),
            template='{"story": "{{story}}"}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Use appropriate max_tokens**: Set based on expected output length
2. **Adjust temperature**: 0.7 for factual content, 0.9 for creative writing
3. **Add deduplication**: Always use `.check_hash()` or `.check_simhash()`
4. **Language filtering**: Use `.check_language()` for multilingual datasets
5. **Worker count**: Set based on API rate limits (4-8 for OpenAI)
6. **Template organization**: Use external Jinja2 files for complex prompts
7. **Error handling**: Check for API keys before running
8. **Output structure**: Use JSON for structured data storage

## Reference

For more examples, see:
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_basic.py`
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_steps.py`
