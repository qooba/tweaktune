"""
Reinforcement Learning Dataset Formats

Demonstrates how to generate datasets for various RL fine-tuning methods:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)

Based on test_steps.py render_sft, render_dpo, render_grpo tests.
"""

from tweaktune import Pipeline
import json


def main():
    print("=" * 70)
    print("Example 1: Supervised Fine-Tuning (SFT) Format")
    print("=" * 70)
    print("Standard conversation format for supervised fine-tuning.\n")

    (Pipeline()
        .with_workers(1)
        .iter_range(3)
            # Create conversation components
            .add_column("system", lambda data: "You are a helpful assistant.")
            .add_column("question", lambda data: "Hello, who won the world series in 2020?")
            .add_column("call1", lambda data: {"name": "get_who_won", "arguments": {"year": 2020}})
            .add_column("response", lambda data: '{"winner": "Los Angeles Dodgers", "year": 2020}')
            .add_column("answer", lambda data: "The Los Angeles Dodgers won the World Series in 2020.")

            # Render as SFT format
            .render_sft(
                conversation="@s:system|@u:question|@a:tool_calls([call1])|@t:response|@a:answer",
                output="conversation"
            )

            .write_jsonl(path="15_sft_format.jsonl", value="conversation")
        .run())

    print("Written to 15_sft_format.jsonl")

    # Display sample
    with open("15_sft_format.jsonl", "r") as f:
        sample = json.loads(f.readline())
        print("\nSample SFT conversation structure:")
        print(f"  Number of messages: {len(sample['messages'])}")
        for i, msg in enumerate(sample['messages'], 1):
            role = msg['role']
            if 'tool_calls' in msg:
                print(f"  {i}. {role}: [tool_calls]")
            elif 'content' in msg:
                content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                print(f"  {i}. {role}: {content}")

    print("\n" + "=" * 70)
    print("Example 2: Direct Preference Optimization (DPO) Format")
    print("=" * 70)
    print("Conversation with chosen and rejected responses for preference learning.\n")

    (Pipeline()
        .with_workers(1)
        .iter_range(5)
            # Create conversation components
            .add_column("system", lambda data: "You are a helpful assistant.")
            .add_column("question", lambda data: f"Who won the world series in {2015 + data['index']}?")

            # Create chosen (correct) and rejected (incorrect) responses
            .add_column("call_chosen", lambda data: {
                "name": "get_who_won",
                "arguments": {"year": 2015 + data['index']}
            })
            .add_column("call_rejected", lambda data: {
                "name": "get_who_won",
                "arguments": {"year": 2015 + data['index'] + 1}  # Wrong year
            })

            # Render as DPO format
            .render_dpo(
                conversation="@s:system|@u:question",
                chosen="call_chosen",
                rejected="call_rejected",
                output="conversation"
            )

            .write_jsonl(path="15_dpo_format.jsonl", value="conversation")
        .run())

    print("Written to 15_dpo_format.jsonl")

    # Display sample
    with open("15_dpo_format.jsonl", "r") as f:
        sample = json.loads(f.readline())
        print("\nSample DPO conversation structure:")
        print(f"  Messages: {len(sample['messages'])} (context only)")
        print(f"  Has 'chosen' field: {'chosen' in sample}")
        print(f"  Has 'rejected' field: {'rejected' in sample}")
        if 'chosen' in sample:
            chosen_preview = sample['chosen'][:60] + "..." if len(sample['chosen']) > 60 else sample['chosen']
            print(f"  Chosen preview: {chosen_preview}")
        if 'rejected' in sample:
            rejected_preview = sample['rejected'][:60] + "..." if len(sample['rejected']) > 60 else sample['rejected']
            print(f"  Rejected preview: {rejected_preview}")

    print("\n" + "=" * 70)
    print("Example 3: Group Relative Policy Optimization (GRPO) Format")
    print("=" * 70)
    print("Conversation with solution and validator for RL training.\n")

    (Pipeline()
        .with_workers(1)
        .iter_range(4)
            # Create conversation components
            .add_column("system", lambda data: "You are a helpful assistant that can use tools.")
            .add_column("question", lambda data: f"What's the weather in city #{data['index'] + 1}?")

            # Create solution
            .add_column("solution", lambda data: {
                "name": "get_weather",
                "arguments": {"city_id": data['index'] + 1}
            })

            # Render as GRPO format
            .render_grpo(
                conversation="@s:system|@u:question",
                solution="solution",
                validator_id="tool_use",
                output="conversation"
            )

            .write_jsonl(path="15_grpo_format.jsonl", value="conversation")
        .run())

    print("Written to 15_grpo_format.jsonl")

    # Display sample
    with open("15_grpo_format.jsonl", "r") as f:
        sample = json.loads(f.readline())
        print("\nSample GRPO conversation structure:")
        print(f"  Messages: {len(sample['messages'])} (context only)")
        print(f"  Has 'solution' field: {'solution' in sample}")
        print(f"  Has 'validator_id' field: {'validator_id' in sample}")
        if 'solution' in sample:
            solution_preview = sample['solution'][:70] + "..." if len(sample['solution']) > 70 else sample['solution']
            print(f"  Solution preview: {solution_preview}")
        if 'validator_id' in sample:
            print(f"  Validator ID: {sample['validator_id']}")

    print("\n" + "=" * 70)
    print("Example 4: Multi-turn DPO with Conversations")
    print("=" * 70)
    print("More complex DPO example with conversation history.\n")

    (Pipeline()
        .with_workers(1)
        .iter_range(2)
            .add_column("system", lambda data: "You are a math tutor.")
            .add_column("q1", lambda data: "What is 5 + 3?")
            .add_column("a1", lambda data: "5 + 3 equals 8.")
            .add_column("q2", lambda data: "What about 12 - 4?")

            # Good response (detailed)
            .add_column("good_answer", lambda data: {
                "content": "12 - 4 equals 8. Notice that both 5+3 and 12-4 give us the same result!"
            })

            # Bad response (minimal)
            .add_column("bad_answer", lambda data: {
                "content": "8"
            })

            .render_dpo(
                conversation="@s:system|@u:q1|@a:a1|@u:q2",
                chosen="good_answer",
                rejected="bad_answer",
                output="conversation"
            )

            .write_jsonl(path="15_dpo_multiturn.jsonl", value="conversation")
        .run())

    print("Written to 15_dpo_multiturn.jsonl")

    # Display sample
    with open("15_dpo_multiturn.jsonl", "r") as f:
        sample = json.loads(f.readline())
        print("\nSample multi-turn DPO structure:")
        print(f"  Context messages: {len(sample['messages'])}")
        for i, msg in enumerate(sample['messages'], 1):
            content = msg.get('content', '')[:40] + "..." if len(msg.get('content', '')) > 40 else msg.get('content', '')
            print(f"  {i}. {msg['role']}: {content}")

        # Parse chosen/rejected (they are JSON strings in this case)
        chosen_data = json.loads(sample['chosen'])
        rejected_data = json.loads(sample['rejected'])
        print(f"\n  Chosen response: {chosen_data['content']}")
        print(f"  Rejected response: {rejected_data['content']}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Three RL dataset formats have been generated:

1. SFT (15_sft_format.jsonl)
   - Standard conversation format
   - Complete conversation with all messages
   - Used for supervised fine-tuning

2. DPO (15_dpo_format.jsonl, 15_dpo_multiturn.jsonl)
   - Context messages + chosen/rejected pairs
   - Used for preference learning
   - Trains model to prefer chosen over rejected

3. GRPO (15_grpo_format.jsonl)
   - Context messages + solution + validator_id
   - Used for group relative policy optimization
   - Includes ground truth solution and validation method

All formats support tool calling and multi-turn conversations.
    """)


if __name__ == "__main__":
    main()
