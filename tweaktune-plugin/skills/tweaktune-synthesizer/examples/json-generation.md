# JSON/Structured Data Generation Examples

This document provides examples of generating structured JSON data using tweaktune with Pydantic models.

## Basic JSON Generation

Generate structured personas with Pydantic schema:

```python
from pydantic import BaseModel, Field
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

class PersonaSchema(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Age in years", ge=18, le=100)
    occupation: str = Field(description="Current occupation")
    background: str = Field(description="Brief background story (2-3 sentences)")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/personas.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="persona-generation", metadata=Metadata(description="Generate personas"))
        .with_workers(4)
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("prompt", "Generate a detailed persona for a fictional character")
        .iter_range(100)  # Generate 100 personas
        .generate_structured(
            template="prompt",
            llm="gpt4",
            output="persona",
            response_format=PersonaSchema
        )
        .validate_json(
            schema=PersonaSchema.model_json_schema(),
            instance="persona"
        )
        .write_jsonl(path=str(output_path), value="persona")
        .run()
    )

if __name__ == "__main__":
    main()
```

**Output (output/personas.jsonl):**
```json
{"name": "Sarah Chen", "age": 34, "occupation": "Software Engineer", "background": "Sarah grew up in San Francisco..."}
{"name": "Marcus Johnson", "age": 45, "occupation": "Teacher", "background": "Marcus has been teaching..."}
```

## Complex Nested JSON

Generate complex nested structures:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class Skill(BaseModel):
    name: str
    level: str = Field(description="beginner, intermediate, expert")
    years_experience: int

class Profile(BaseModel):
    name: str
    email: str
    age: int = Field(ge=18, le=100)
    address: Address
    skills: List[Skill] = Field(min_items=2, max_items=5)
    bio: str
    website: Optional[str] = None

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/profiles.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="profile-generation", metadata=Metadata(description="Generate complex profiles"))
        .with_workers(4)
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("prompt", "Generate a complete professional profile for a fictional person")
        .iter_range(50)
        .generate_structured(
            template="prompt",
            llm="gpt4",
            output="profile",
            response_format=Profile
        )
        .validate_json(
            schema=Profile.model_json_schema(),
            instance="profile"
        )
        .write_jsonl(path=str(output_path), value="profile")
        .run()
    )

if __name__ == "__main__":
    main()
```

## JSON with Context

Generate JSON based on seed data:

```python
from pydantic import BaseModel, Field
from typing import List
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

class ArticleSummary(BaseModel):
    title: str = Field(description="Article title")
    key_points: List[str] = Field(min_items=3, max_items=5, description="Main points")
    sentiment: str = Field(description="positive, negative, or neutral")
    category: str = Field(description="Article category")
    word_count_estimate: int = Field(description="Estimated word count")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/summaries.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="summary-generation", metadata=Metadata(description="Generate article summaries"))
        .with_workers(4)
        .with_jsonl_dataset("articles", "articles.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("prompt", "Analyze and summarize this article:\n\n{{article_text}}")
        .iter_dataset("articles")
        .generate_structured(
            template="prompt",
            llm="gpt4",
            output="summary",
            response_format=ArticleSummary
        )
        .validate_json(
            schema=ArticleSummary.model_json_schema(),
            instance="summary"
        )
        .write_jsonl(
            path=str(output_path),
            template='{"article_id": "{{article_id}}", "summary": {{summary|tojson}}}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## Using generate_json (Alternative Method)

Generate JSON using `.generate_json()` with json_path:

```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/entities.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="entity-extraction", metadata=Metadata(description="Extract entities"))
        .with_workers(4)
        .with_jsonl_dataset("texts", "texts.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template(
            "prompt",
            "Extract entities from this text in JSON format:\n\n{{text}}\n\n"
            "Provide: {\"people\": [], \"organizations\": [], \"locations\": []}"
        )
        .iter_dataset("texts")
        .generate_json(
            template="prompt",
            llm="gpt4",
            output="entities",
            json_path="$"
        )
        .check_json("entities")  # Validate JSON structure
        .write_jsonl(
            path=str(output_path),
            template='{"text_id": "{{text_id}}", "entities": {{entities|tojson}}}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## Classification/Labeling

Generate labels and classifications:

```python
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

class SentimentEnum(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class CategoryEnum(str, Enum):
    tech = "technology"
    sports = "sports"
    politics = "politics"
    entertainment = "entertainment"
    science = "science"

class TextClassification(BaseModel):
    sentiment: SentimentEnum
    category: CategoryEnum
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: List[str] = Field(min_items=1, max_items=5)

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/classifications.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="text-classification", metadata=Metadata(description="Classify texts"))
        .with_workers(4)
        .with_jsonl_dataset("texts", "texts.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template(
            "prompt",
            "Classify this text:\n\n{{text}}\n\n"
            "Provide sentiment, category, confidence score, and keywords."
        )
        .iter_dataset("texts")
        .generate_structured(
            template="prompt",
            llm="gpt4",
            output="classification",
            response_format=TextClassification
        )
        .validate_json(
            schema=TextClassification.model_json_schema(),
            instance="classification"
        )
        .write_jsonl(
            path=str(output_path),
            template='{"text": "{{text}}", "classification": {{classification|tojson}}}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## Multi-field JSON Generation

Generate multiple JSON fields in sequence:

```python
from pydantic import BaseModel, Field
from typing import List
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

class Question(BaseModel):
    question: str
    difficulty: str = Field(description="easy, medium, hard")

class Answer(BaseModel):
    answer: str
    explanation: str

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/qa_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="qa-generation", metadata=Metadata(description="Generate Q&A pairs"))
        .with_workers(4)
        .with_jsonl_dataset("topics", "topics.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("question_prompt", "Generate a question about: {{topic}}")
        .with_template("answer_prompt", "Answer this question: {{question_obj.question}}")
        .iter_dataset("topics")
        # Generate question
        .generate_structured(
            template="question_prompt",
            llm="gpt4",
            output="question_obj",
            response_format=Question
        )
        # Generate answer
        .generate_structured(
            template="answer_prompt",
            llm="gpt4",
            output="answer_obj",
            response_format=Answer
        )
        .validate_json(schema=Question.model_json_schema(), instance="question_obj")
        .validate_json(schema=Answer.model_json_schema(), instance="answer_obj")
        .write_jsonl(
            path=str(output_path),
            template='{"topic": "{{topic}}", "question": {{question_obj|tojson}}, "answer": {{answer_obj|tojson}}}'
        )
        .run()
    )

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Use Pydantic models**: Define clear schemas with Field descriptions
2. **Add validation constraints**: Use `ge`, `le`, `min_items`, `max_items`, etc.
3. **Validate generated JSON**: Always use `.validate_json()` or `.check_json()`
4. **Use enums for categories**: Define allowed values with Enum
5. **Nested structures**: Break down complex objects into smaller models
6. **Temperature**: Use lower temperature (0.5-0.7) for structured data
7. **Field descriptions**: Provide clear descriptions for better generation
8. **Error handling**: Validate schemas before writing to output

## Reference

For more examples, see:
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_steps.py`
