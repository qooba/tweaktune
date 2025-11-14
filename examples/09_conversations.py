"""
Conversation Dataset Generation

Creates multi-turn conversation datasets in OpenAI format.
Based on test_steps.py render_conversation tests.
"""

from tweaktune import Pipeline
import json

def main():
    print("Example 1: Simple conversation format")
    (Pipeline()
        .with_workers(1)
        .iter_range(5)
            .add_column("system", lambda data: "You are a helpful assistant.")
            .add_column("user_msg", lambda data: f"Hello! This is message {data['index']}")
            .add_column("assistant_msg", lambda data: f"Hi! Thanks for message {data['index']}")

            # Build conversation
            .render_conversation(
                conversation="""@system:system|@user:user_msg|@assistant:assistant_msg""",
                output="conversation"
            )

            .write_jsonl(path="09_simple_conv.jsonl", value="conversation")
        .run())
    print("Written to 09_simple_conv.jsonl\n")

    print("Example 2: Multi-turn conversation")
    (Pipeline()
        .with_workers(1)
        .iter_range(3)
            .add_column("system", lambda data: "You are a math tutor.")
            .add_column("q1", lambda data: "What is 2 + 2?")
            .add_column("a1", lambda data: "2 + 2 equals 4.")
            .add_column("q2", lambda data: "What about 3 + 5?")
            .add_column("a2", lambda data: "3 + 5 equals 8.")
            .add_column("thanks", lambda data: "Thank you!")
            .add_column("welcome", lambda data: "You're welcome! Feel free to ask more questions.")

            .render_conversation(
                conversation="""
@system:system
@user:q1
@assistant:a1
@user:q2
@assistant:a2
@user:thanks
@assistant:welcome
                """,
                separator="\n",  # Use newline as separator
                output="conversation"
            )

            .write_jsonl(path="09_multiturn.jsonl", value="conversation")
        .run())
    print("Written to 09_multiturn.jsonl\n")

    print("Example 3: Conversation with role aliases")
    (Pipeline()
        .with_workers(1)
        .iter_range(3)
            .add_column("sys", lambda data: "You are a creative writing assistant.")
            .add_column("request", lambda data: "Help me write a story opening.")
            .add_column("response", lambda data: "I'd be happy to help! What genre interests you?")
            .add_column("genre", lambda data: "Science fiction.")
            .add_column("story", lambda data: "Great choice! Here's an opening: 'The stars had never seemed so distant...'")

            # Use short aliases: @s = system, @u = user, @a = assistant
            .render_conversation(
                conversation="""
@s:sys
@u:request
@a:response
@u:genre
@a:story
                """,
                separator="\n",
                output="conversation"
            )

            .write_jsonl(path="09_aliases.jsonl", value="conversation")
        .run())
    print("Written to 09_aliases.jsonl\n")

    print("Example 4: Conversation with reasoning")
    (Pipeline()
        .with_workers(1)
        .iter_range(2)
            .add_column("system", lambda data: "You are a problem-solving assistant.")
            .add_column("problem", lambda data: "How can I optimize my code?")
            .add_column("thinking", lambda data: "Let me analyze the problem step by step...")
            .add_column("solution", lambda data: "Here are three optimization strategies...")

            .render_conversation(
                conversation="""
@system:system
@user:problem
@assistant:think(thinking)
@assistant:solution
                """,
                output="conversation",
                separator="\n"
            )

            .write_jsonl(path="09_reasoning.jsonl", value="conversation")
        .run())
    print("Written to 09_reasoning.jsonl\n")

    # Display sample
    print("Sample multi-turn conversation:")
    with open("09_multiturn.jsonl", "r") as f:
        first = json.loads(f.readline())
        messages = first["messages"]
        for msg in messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            if content:
                print(f"  [{role}] {content}")

    print("\nSample conversation with reasoning:")
    with open("09_reasoning.jsonl", "r") as f:
        first = json.loads(f.readline())
        messages = first["messages"]
        for msg in messages:
            role = msg["role"].upper()
            if "reasoning_content" in msg:
                print(f"  [{role}] (thinking) {msg['reasoning_content']}")
            content = msg.get("content", "")
            if content:
                print(f"  [{role}] {content}")

if __name__ == "__main__":
    main()
