"""
Basic TweakTune Pipeline Template

This is a minimal pipeline template showing the essential structure.
Customize the sections marked with {{PLACEHOLDERS}} based on your needs.
"""

from tweaktune import Pipeline, Metadata
import os
from pathlib import Path


def main():
    # ===== Configuration =====
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")

    # Output configuration
    output_path = Path("output/generated_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Build and Run Pipeline =====
    (Pipeline(
        name="my-pipeline",  # TODO: Change to descriptive name
        metadata=Metadata(description="Pipeline description")  # Optional but recommended
    )
        # Configure workers (adjust based on API rate limits)
        .with_workers(4)

        # ===== Resource Configuration =====
        # TODO: Add your datasets, LLMs, and templates here

        # Example dataset configurations (uncomment and modify as needed):
        # .with_jsonl_dataset("my_data", "input.jsonl")
        # .with_parquet_dataset("my_data", "input.parquet")
        # .with_csv_dataset("my_data", "input.csv", delimiter=",", has_header=True)

        # LLM configuration
        .with_llm_openai("gpt4", api_key, "gpt-4")

        # Template configuration
        # .with_template("my_prompt", "Your prompt template here: {{variable}}")

        # ===== Start Iteration =====
        # TODO: Choose iteration method
        .iter_range(100)  # Generate from scratch (100 examples)
        # .iter_dataset("my_data")  # Or iterate over existing dataset

        # ===== Pipeline Steps =====
        # TODO: Add your pipeline steps here

        # Example steps (uncomment and modify as needed):
        # .sample(dataset="my_data", size=1, output="sampled")
        # .generate_text(template="my_prompt", llm="gpt4", output="result", max_tokens=1024)
        # .add_column("new_field", lambda data: process(data))
        # .check_hash("result")  # Deduplication

        # ===== Output =====
        # Write results to file
        .write_jsonl(
            path=str(output_path),
            template='{"result": "{{result}}"}'  # TODO: Customize output format
        )

        # ===== Execute =====
        .run()  # Or use .ui() for web interface
    )

    print(f"Pipeline completed! Output written to: {output_path}")


if __name__ == "__main__":
    main()
