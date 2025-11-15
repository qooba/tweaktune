"""
JSON Generation Example

Demonstrates structured JSON generation with LLMs.
Requires OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python 06_json_generation.py
"""

from tweaktune import Pipeline
from pydantic import BaseModel, Field
import os
import json

class Person(BaseModel):
    """Person profile"""
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    occupation: str = Field(..., description="Job title or profession")
    hobbies: list[str] = Field(..., description="List of hobbies")

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("Example 1: Generate JSON with json_path")
    (Pipeline()
        .with_workers(2)
        .with_llm_openai("gpt4", api_key, "gpt-4o-mini")
        .with_template("prompt", """
Generate a fictional person profile.
Return JSON: {"person": {"name": "...", "age": ..., "occupation": "...", "city": "..."}}
""")
        .with_template("output", """{"profile": {{person|tojson}}}""")
        .iter_range(3)
            .generate_json(
                template="prompt",
                llm="gpt4",
                output="person",
                json_path="person",  # Extract the "person" field
                max_tokens=150,
                temperature=0.8
            )
            .write_jsonl(path="06_profiles.jsonl", template="output")
        .run())
    print("Written to 06_profiles.jsonl\n")

    print("Example 2: Structured output with Pydantic")
    (Pipeline()
        .with_workers(2)
        .with_llm_openai("gpt4", api_key, "gpt-4o-mini")
        .with_template("prompt", "Generate a fictional person profile")
        .with_template("output", """{"person": {{person|tojson}}}""")
        .iter_range(3)
            .generate_structured(
                template="prompt",
                llm="gpt4",
                output="person",
                response_format=Person,
                max_tokens=200,
                temperature=0.8
            )
            .write_jsonl(path="06_structured.jsonl", template="output")
        .run())
    print("Written to 06_structured.jsonl\n")

    print("Example 3: Extract specific fields")
    (Pipeline()
        .with_workers(2)
        .with_llm_openai("gpt4", api_key, "gpt-4o-mini")
        .with_template("prompt", """
Analyze this topic: {{topic}}
Return JSON: {
  "analysis": {
    "complexity": "low|medium|high",
    "main_concepts": ["concept1", "concept2"],
    "applications": ["app1", "app2"]
  }
}
""")
        .with_template("output", """
{
  "topic": {{topic|jstr}},
  "complexity": {{complexity|jstr}},
  "concepts": {{concepts|tojson}},
  "applications": {{applications|tojson}}
}
""")
        .iter_range(3)
            .add_column("topic", lambda data: [
                "machine learning",
                "blockchain technology",
                "quantum physics"
            ][data["index"]])
            .generate_json(
                template="prompt",
                llm="gpt4",
                output="analysis",
                json_path="analysis",
                max_tokens=200
            )
            # Extract nested fields
            .add_column("complexity", lambda data: data["analysis"]["complexity"])
            .add_column("concepts", lambda data: data["analysis"]["main_concepts"])
            .add_column("applications", lambda data: data["analysis"]["applications"])
            .write_jsonl(path="06_analysis.jsonl", template="output")
        .run())
    print("Written to 06_analysis.jsonl\n")

    # Show results
    print("Structured profiles:")
    with open("06_structured.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            person = data["person"]
            print(f"  {person['name']}, {person['age']}, {person['occupation']}")
            print(f"    Hobbies: {', '.join(person['hobbies'])}")

if __name__ == "__main__":
    main()
