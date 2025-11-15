"""
Question-Answer Dataset Generation

Creates a complete Q&A dataset using LLMs.
Requires OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python 07_qa_dataset.py
"""

from tweaktune import Pipeline
import os
import json

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    topics = [
        "Python programming",
        "Data structures",
        "Machine learning basics",
        "Web development",
        "Database design",
        "Cloud computing",
        "Cybersecurity",
        "DevOps practices"
    ]

    print("Generating Q&A dataset...")
    (Pipeline()
        .with_workers(3)
        .with_llm_openai(
            name="gpt4",
            api_key=api_key,
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0.7
        )

        # Templates
        .with_template("system", "You are an expert educator creating training materials.")
        .with_template("question_prompt", "Generate an intermediate-level question about: {{topic}}")
        .with_template("answer_prompt", "Answer this question concisely but completely: {{question}}")
        .with_template("output", """{"topic": {{topic|jstr}},"question": {{question|jstr}},"answer": {{answer|jstr}}}""")

        .iter_range(len(topics))
            # Select topic
            .add_column("topic", lambda data: topics[data["index"]])

            # Generate question
            .generate_text(
                template="question_prompt",
                llm="gpt4",
                output="question",
                system_template="system",
                max_tokens=100,
                temperature=0.8
            )

            # Generate answer
            .generate_text(
                template="answer_prompt",
                llm="gpt4",
                output="answer",
                system_template="system",
                max_tokens=300,
                temperature=0.7
            )

            .write_jsonl(path="07_qa_dataset.jsonl", template="output")
        .run())

    print("Generated Q&A dataset\n")

    # Display results
    print("Sample Q&A pairs:")
    with open("07_qa_dataset.jsonl", "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            print(f"\n{i+1}. Topic: {data['topic']}")
            print(f"   Q: {data['question']}")
            print(f"   A: {data['answer'][:100]}...")

if __name__ == "__main__":
    main()
