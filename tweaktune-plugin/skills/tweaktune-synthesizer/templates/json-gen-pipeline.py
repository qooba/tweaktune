"""
JSON Generation Pipeline Template

Generates structured JSON data using Pydantic models.
"""

from pydantic import BaseModel, Field
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path


# ===== Define Pydantic Schema =====
class PersonaSchema(BaseModel):
    """Schema for persona data"""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Age in years", ge=18, le=100)
    occupation: str = Field(description="Current occupation")
    background: str = Field(description="Brief background story (2-3 sentences)")
    skills: list[str] = Field(description="List of 3-5 skills", min_items=3, max_items=5)


def main():
    # ===== Configuration =====
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/generated_personas.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Build and Run Pipeline =====
    (Pipeline(
        name="json-generation-pipeline",
        metadata=Metadata(description="Generate structured JSON data")
    )
        # Worker configuration
        .with_workers(4)

        # ===== Resource Configuration =====
        # LLM: OpenAI GPT-4
        .with_llm_openai("gpt4", api_key, "gpt-4")

        # Template: Define the generation prompt
        .with_template(
            "persona_prompt",
            """Generate a detailed, realistic persona for a fictional character.
Create a unique and diverse character with interesting background."""
        )

        # ===== Start Iteration =====
        .iter_range(100)  # Generate 100 personas from scratch

        # ===== Pipeline Steps =====

        # Step 1: Generate structured JSON using Pydantic schema
        .generate_structured(
            template="persona_prompt",
            llm="gpt4",
            output="persona",
            response_format=PersonaSchema  # Ensures JSON matches schema
        )

        # Step 2: Validate generated JSON against schema
        .validate_json(
            schema=PersonaSchema.model_json_schema(),
            instance="persona"
        )

        # Step 3: Quality check - deduplication by name
        .check_hash("persona.name")

        # ===== Output =====
        .write_jsonl(
            path=str(output_path),
            value="persona"  # Write the persona object directly
        )

        # ===== Execute =====
        .run()
    )

    print(f"✓ JSON generation completed!")
    print(f"✓ Generated {100} personas")
    print(f"✓ Output written to: {output_path}")


if __name__ == "__main__":
    main()
