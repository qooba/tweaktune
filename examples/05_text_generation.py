"""
Text Generation Example

Demonstrates LLM integration for text generation.
Requires OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python 05_text_generation.py
"""

from tweaktune import Pipeline
import os
import json

def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print("Example 1: Simple text generation")
    (Pipeline()
        .with_workers(2)
        .with_llm_openai(
            name="gpt4",
            api_key=api_key,
            model="gpt-4o-mini",  # Use mini for cost efficiency
            max_tokens=100,
            temperature=0.7
        )
        .with_template("system", "You are a creative writer.")
        .with_template("prompt", "Write a one-sentence description of {{topic}}")
        .with_template("output", """
{
  "topic": {{topic|jstr}},
  "description": {{description|jstr}}
}
""")
        .iter_range(5)
            .add_column("topic", lambda data: [
                "artificial intelligence",
                "quantum computing",
                "renewable energy",
                "space exploration",
                "biotechnology"
            ][data["index"]])
            .generate_text(
                template="prompt",
                llm="gpt4",
                output="description",
                system_template="system",
                max_tokens=100,
                temperature=0.7
            )
            .write_jsonl(path="05_descriptions.jsonl", template="output")
        .run())
    print("Written to 05_descriptions.jsonl\n")

    print("Example 2: Question generation")
    (Pipeline()
        .with_workers(2)
        .with_llm_openai(
            name="gpt4",
            api_key=api_key,
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0.8
        )
        .with_template("system", "You are an educational content creator.")
        .with_template("prompt", "Generate an interesting question about {{subject}}")
        .with_template("output", """{"subject": {{subject|jstr}}, "question": {{question|jstr}}}""")
        .iter_range(5)
            .add_column("subject", lambda data: [
                "mathematics",
                "history",
                "physics",
                "literature",
                "biology"
            ][data["index"]])
            .generate_text(
                template="prompt",
                llm="gpt4",
                output="question",
                system_template="system",
                max_tokens=50,
                temperature=0.8
            )
            .write_jsonl(path="05_questions.jsonl", template="output")
        .run())
    print("Written to 05_questions.jsonl\n")

    # Show results
    print("Generated descriptions:")
    with open("05_descriptions.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            print(f"  {data['topic']}: {data['description']}")

    print("\nGenerated questions:")
    with open("05_questions.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            print(f"  [{data['subject']}] {data['question']}")

if __name__ == "__main__":
    main()
