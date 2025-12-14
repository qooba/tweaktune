"""
Conversation Generation Pipeline Template

Generates multi-turn conversations using the Conv() builder.
"""

from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path


def main():
    # ===== Configuration =====
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Build and Run Pipeline =====
    (Pipeline(
        name="conversation-generation-pipeline",
        metadata=Metadata(description="Generate multi-turn conversations")
    )
        # Worker configuration
        .with_workers(4)

        # ===== Resource Configuration =====
        # Dataset: Topics for conversations
        .with_jsonl_dataset("topics", "topics.jsonl")  # TODO: Update path
        # Expected format: {"topic": "Climate Change"}

        # LLM: OpenAI GPT-4
        .with_llm_openai("gpt4", api_key, "gpt-4")

        # Templates: Define prompts for each turn
        .with_template(
            "question_prompt",
            "Generate a thoughtful question about: {{topic}}"
        )
        .with_template(
            "answer_prompt",
            "Provide a detailed, helpful answer to this question: {{question}}"
        )
        .with_template(
            "followup_prompt",
            "Generate a follow-up question based on this answer: {{answer}}"
        )
        .with_template(
            "final_answer_prompt",
            "Answer the follow-up question: {{followup}}"
        )

        # ===== Start Iteration =====
        .iter_dataset("topics")  # Iterate over topics
        # Alternative: .iter_range(100) for generation without seed data

        # ===== Pipeline Steps =====

        # Step 1: Set system message
        .add_literal(
            "system",
            "You are a knowledgeable and helpful assistant. Provide clear, accurate, and engaging responses."
        )

        # Step 2: Generate first question
        .generate_text(
            template="question_prompt",
            llm="gpt4",
            output="question",
            max_tokens=100,
            temperature=0.8
        )

        # Step 3: Generate first answer
        .generate_text(
            template="answer_prompt",
            llm="gpt4",
            output="answer",
            max_tokens=512,
            temperature=0.7
        )

        # Step 4: Generate follow-up question
        .generate_text(
            template="followup_prompt",
            llm="gpt4",
            output="followup",
            max_tokens=100,
            temperature=0.8
        )

        # Step 5: Generate final answer
        .generate_text(
            template="final_answer_prompt",
            llm="gpt4",
            output="final_answer",
            max_tokens=512,
            temperature=0.7
        )

        # Step 6: Build conversation using Conv() builder
        .render_conversation(
            conversation=Conv()
                .system("system")         # System message
                .user("question")         # First user question
                .assistant("answer")      # First assistant answer
                .user("followup")         # Follow-up question
                .assistant("final_answer"),  # Final answer
            output="conversation"
        )

        # Step 7: Validate conversation format
        .validate_conversation("conversation")

        # ===== Output =====
        .write_jsonl(
            path=str(output_path),
            value="conversation"  # Write conversation object
        )

        # ===== Execute =====
        .run()
    )

    print(f"✓ Conversation generation completed!")
    print(f"✓ Output written to: {output_path}")
    print(f"\nOutput format: OpenAI-compatible messages array")


if __name__ == "__main__":
    main()
