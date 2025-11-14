# LLM Integration

tweaktune integrates with various LLM providers for synthetic data generation.

## OpenAI-Compatible APIs

### OpenAI

```python
import os

(Pipeline()
    .with_llm_openai(
        name="gpt4",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4",
        max_tokens=2048,
        temperature=0.7
    )
    .iter_range(10)
        .add_column("question", lambda data: "What is AI?")
        .generate_text(
            template="question",
            llm="gpt4",
            output="answer"
        )
        .write_jsonl(path="qa.jsonl", value="answer")
    .run())
```

### Azure OpenAI

```python
.with_llm_azure_openai(
    name="azure_gpt",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    endpoint="https://your-resource.openai.azure.com",
    deployment_name="gpt-4",
    api_version="2024-02-15-preview",
    max_tokens=2048,
    temperature=0.7
)
```

### Custom OpenAI-Compatible API

For local models or other OpenAI-compatible endpoints:

```python
.with_llm_api(
    name="local_model",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Some local servers don't need keys
    model="mistral-7b",
    max_tokens=1024,
    temperature=0.7
)
```

## Text Generation

Generate unstructured text:

```python
.with_template("prompt", """Write a short product description for: {{product_name}}""")

.iter_range(10)
    .add_column("product_name", lambda data: f"Product {data['index']}")
    .generate_text(
        template="prompt",
        llm="gpt4",
        output="description",
        max_tokens=100,
        temperature=0.8
    )
```

With system prompt:

```python
.with_template("system", """You are a creative marketing copywriter.""")
.with_template("prompt", """Write a catchy slogan for: {{product}}""")

.generate_text(
    template="prompt",
    llm="gpt4",
    output="slogan",
    system_template="system",
    max_tokens=50,
    temperature=0.9
)
```

## JSON Generation

Generate structured JSON output:

```python
.with_template("prompt", """
Generate a person's profile with name, age, and occupation for: {{seed}}
""")

.iter_range(10)
    .add_column("seed", lambda data: f"person_{data['index']}")
    .generate_json(
        template="prompt",
        llm="gpt4",
        output="profile",
        json_path="profile",  # Extract specific field from response
        max_tokens=200,
        temperature=0.7
    )
```

The `json_path` parameter extracts a specific field from the LLM's JSON response.

## Structured Output with Pydantic

Generate data conforming to a Pydantic schema:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    occupation: str = Field(..., description="Job title")
    bio: str = Field(..., description="Short biography")

.with_template("prompt", "Generate a fictional person profile")

.generate_structured(
    template="prompt",
    llm="gpt4",
    output="person",
    response_format=Person,
    max_tokens=300,
    temperature=0.7
)
```

This uses OpenAI's structured output feature (requires compatible model).

## Schema Template

For dynamic schemas, use a schema template:

```python
.with_template("schema_template", """
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "category": {"type": "string", "enum": ["A", "B", "C"]},
    "score": {"type": "number", "minimum": 0, "maximum": 100}
  },
  "required": ["title", "category", "score"]
}
""")

.generate_json(
    template="prompt",
    llm="gpt4",
    output="result",
    schema_template="schema_template",
    max_tokens=200
)
```

## Local Models

### Unsloth

For local inference with Unsloth:

```python
# Requires: pip install unsloth

.with_llm_unsloth(
    name="local_llm",
    model_name="unsloth/mistral-7b-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=2048,
    hf_token=os.environ.get("HF_TOKEN"),
    chat_template="chatml",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
)
```

### MistralRS

For MistralRS integration:

```python
# Requires: pip install mistralrs

.with_llm_mistralrs(
    name="mistral",
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    in_situ_quant="Q4K"
)
```

## Temperature and Randomness

Control output randomness with temperature:

- `temperature=0.1` - More deterministic, focused
- `temperature=0.7` - Balanced (default)
- `temperature=1.0` - More random, creative

```python
# Deterministic answers
.generate_text(..., temperature=0.1)

# Creative writing
.generate_text(..., temperature=0.9)
```

## Max Tokens

Control response length:

```python
# Short responses
.generate_text(..., max_tokens=50)

# Long responses
.generate_text(..., max_tokens=2048)
```

## Complete Example

```python
import os
from tweaktune import Pipeline

(Pipeline()
    .with_workers(5)
    .with_llm_openai(
        name="gpt4",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4",
        max_tokens=1024,
        temperature=0.7
    )
    .with_template("system", "You are a helpful assistant creating training data.")
    .with_template("prompt", """
Generate a question and answer pair about: {{topic}}
Return JSON: {"question": "...", "answer": "..."}
""")
    .with_template("output", """
{
  "topic": {{topic|jstr}},
  "qa": {{qa_pair|tojson}}
}
""")
    .iter_range(100)
        .add_column("topic", lambda data: f"Topic {data['index']}")
        .generate_json(
            template="prompt",
            llm="gpt4",
            output="qa_pair",
            system_template="system",
            max_tokens=200,
            temperature=0.7
        )
        .write_jsonl(path="qa_dataset.jsonl", template="output")
    .run())
```

## Best Practices

1. **Use structured output** when you need consistent JSON format
2. **Set appropriate max_tokens** to avoid incomplete responses
3. **Adjust temperature** based on your use case
4. **Use system prompts** to guide model behavior
5. **Add retries** for production pipelines (see Custom Steps)
6. **Monitor costs** when using paid APIs

## Next Steps

- Learn about [Pipeline Steps](06-pipeline-steps.md)
- Explore [Conversation & Tools](09-conversation-tools.md) for function calling
- See [examples](/examples) for complete use cases
