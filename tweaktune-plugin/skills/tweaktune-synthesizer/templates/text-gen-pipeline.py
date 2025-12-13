"""
Text Generation Pipeline Template

Generates text content from topics/prompts with quality checks.
"""

from tweaktune import Pipeline, Metadata
import os
from pathlib import Path


def main():
    # ===== Configuration =====
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/generated_text.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Build and Run Pipeline =====
    (Pipeline(
        name="text-generation-pipeline",
        metadata=Metadata(description="Generate text content with quality checks")
    )
        # Worker configuration (adjust based on API rate limits)
        .with_workers(4)

        # ===== Resource Configuration =====
        # Dataset: Load topics to generate text about
        .with_jsonl_dataset("topics", "topics.jsonl")  # TODO: Update path
        # Expected format: {"topic": "Machine Learning"}

        # LLM: OpenAI GPT-4
        .with_llm_openai("gpt4", api_key, "gpt-4")

        # Template: Define the generation prompt
        .with_template(
            "generation_prompt",
            """Generate a detailed, informative article about the following topic:

Topic: {{topic}}

Please write a comprehensive article that:
- Is well-structured with clear paragraphs
- Is at least 300 words
- Uses clear, engaging language
- Provides accurate information

Article:"""
        )

        # ===== Start Iteration =====
        .iter_dataset("topics")  # Iterate over topics dataset
        # Alternative: .iter_range(100) to generate without seed data

        # ===== Pipeline Steps =====

        # Step 1: Generate text content
        .generate_text(
            template="generation_prompt",
            llm="gpt4",
            output="generated_article",
            max_tokens=2048,  # Adjust based on expected length
            temperature=0.7   # 0.7 for balanced creativity
        )

        # Step 2: Quality checks
        # Deduplication - remove exact duplicates
        .check_hash("generated_article")

        # Fuzzy deduplication - remove near-duplicates
        .check_simhash("generated_article", threshold=0.95)

        # Language filtering - ensure English content
        .check_language(
            input="generated_article",
            language="english",
            precision=0.9  # 90% confidence threshold
        )

        # ===== Output =====
        .write_jsonl(
            path=str(output_path),
            template='{"topic": "{{topic}}", "article": "{{generated_article}}"}'
        )

        # ===== Execute =====
        .run()
        # Alternative: .ui(host="0.0.0.0", port=8080) for web interface
    )

    print(f"✓ Text generation completed!")
    print(f"✓ Output written to: {output_path}")


if __name__ == "__main__":
    main()
