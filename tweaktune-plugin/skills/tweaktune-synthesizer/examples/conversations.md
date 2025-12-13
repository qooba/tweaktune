# Conversation Synthesis Examples

This document provides examples of generating multi-turn conversations using tweaktune's Conv() builder.

## Basic Conversation (Conv() Builder)

Generate simple Q&A conversations:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="conversation-synthesis", metadata=Metadata(description="Generate conversations"))
        .with_workers(4)
        .with_jsonl_dataset("topics", "topics.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("question_prompt", "Generate a question about: {{topic}}")
        .with_template("answer_prompt", "Answer this question: {{question}}")
        .iter_dataset("topics")
        # Add system message
        .add_column("system", lambda data: "You are a helpful assistant.")
        # Generate question
        .generate_text(template="question_prompt", llm="gpt4", output="question")
        # Generate answer
        .generate_text(template="answer_prompt", llm="gpt4", output="answer")
        # Build conversation using Conv() builder
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("question")
                .assistant("answer"),
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

**Output Format (OpenAI compatible):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

## Multi-turn Conversations

Generate complex multi-turn dialogues:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/multi_turn_conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="multi-turn-conversations", metadata=Metadata(description="Multi-turn dialogues"))
        .with_workers(4)
        .with_jsonl_dataset("scenarios", "scenarios.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("q1_prompt", "Generate first question about: {{scenario}}")
        .with_template("a1_prompt", "Answer: {{question1}}")
        .with_template("q2_prompt", "Generate follow-up question based on: {{answer1}}")
        .with_template("a2_prompt", "Answer the follow-up: {{question2}}")
        .iter_dataset("scenarios")
        .add_column("system", lambda data: "You are a knowledgeable assistant.")
        # Turn 1
        .generate_text(template="q1_prompt", llm="gpt4", output="question1")
        .generate_text(template="a1_prompt", llm="gpt4", output="answer1")
        # Turn 2
        .generate_text(template="q2_prompt", llm="gpt4", output="question2")
        .generate_text(template="a2_prompt", llm="gpt4", output="answer2")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("question1")
                .assistant("answer1")
                .user("question2")
                .assistant("answer2"),
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## Conversations with Thinking/Reasoning

Add reasoning content to conversations:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/reasoning_conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="reasoning-conversations", metadata=Metadata(description="Conversations with reasoning"))
        .with_workers(4)
        .with_jsonl_dataset("problems", "problems.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("thinking_prompt", "Think step by step about: {{problem}}")
        .with_template("answer_prompt", "Based on your reasoning, provide final answer to: {{problem}}")
        .iter_dataset("problems")
        .add_column("system", lambda data: "You are a problem-solving assistant.")
        # Generate reasoning
        .generate_text(template="thinking_prompt", llm="gpt4", output="thinking")
        # Generate final answer
        .generate_text(template="answer_prompt", llm="gpt4", output="answer")
        # Build conversation with thinking
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("problem")
                .think("thinking")  # Reasoning content
                .assistant("answer"),
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

**Output includes reasoning_content:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a problem-solving assistant."},
    {"role": "user", "content": "Solve this problem..."},
    {"role": "assistant", "reasoning_content": "Let me think step by step...", "content": "The answer is..."}
  ]
}
```

## String Format (Alternative to Conv())

Use string format for simple conversations:

```python
from tweaktune import Pipeline, Metadata
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/string_format_conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="string-format-conversations", metadata=Metadata(description="String format"))
        .with_workers(4)
        .with_jsonl_dataset("qa_pairs", "qa_pairs.jsonl")
        .iter_dataset("qa_pairs")
        .add_column("system", lambda data: "You are a helpful assistant.")
        # Use string format: @role:field
        .render_conversation(
            conversation="@system:system|@user:question|@assistant:answer",
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

**String format shortcuts:**
- `@system` or `@s` - System message
- `@user` or `@u` - User message
- `@assistant` or `@a` - Assistant message
- `@tool` or `@t` - Tool response

## SFT (Supervised Fine-Tuning) Format

Generate conversations in SFT format:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/sft_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="sft-data-generation", metadata=Metadata(description="SFT format data"))
        .with_workers(4)
        .with_jsonl_dataset("instructions", "instructions.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("response_prompt", "Provide a helpful response to: {{instruction}}")
        .iter_dataset("instructions")
        .add_column("system", lambda data: "You are a helpful AI assistant.")
        # Generate response
        .generate_text(template="response_prompt", llm="gpt4", output="response")
        # Render in SFT format
        .render_sft(
            system="system",
            instruction="instruction",
            response="response",
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## DPO (Direct Preference Optimization) Format

Generate preference pairs for DPO:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/dpo_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="dpo-data-generation", metadata=Metadata(description="DPO format data"))
        .with_workers(4)
        .with_jsonl_dataset("prompts", "prompts.jsonl")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("good_response_prompt", "Provide an excellent response to: {{prompt}}")
        .with_template("bad_response_prompt", "Provide a poor quality response to: {{prompt}}")
        .iter_dataset("prompts")
        .add_column("system", lambda data: "You are an AI assistant.")
        # Generate chosen (good) response
        .generate_text(template="good_response_prompt", llm="gpt4", output="chosen_response")
        # Generate rejected (bad) response
        .generate_text(template="bad_response_prompt", llm="gpt4", output="rejected_response")
        # Render in DPO format
        .render_dpo(
            system="system",
            instruction="prompt",
            chosen="chosen_response",
            rejected="rejected_response",
            output="dpo_pair"
        )
        .write_jsonl(path=str(output_path), value="dpo_pair")
        .run()
    )

if __name__ == "__main__":
    main()
```

**DPO Output Format:**
```json
{
  "chosen": [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Question..."},
    {"role": "assistant", "content": "Excellent response..."}
  ],
  "rejected": [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Question..."},
    {"role": "assistant", "content": "Poor response..."}
  ]
}
```

## Dialogue from Scratch

Generate complete dialogues without seed data:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/dialogues.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="dialogue-from-scratch", metadata=Metadata(description="Generate dialogues"))
        .with_workers(4)
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("topic_prompt", "Generate a random interesting topic")
        .with_template("q_prompt", "Generate a question about: {{topic}}")
        .with_template("a_prompt", "Answer: {{question}}")
        .iter_range(100)
        .add_column("system", lambda data: "You are a helpful assistant.")
        # Generate topic
        .generate_text(template="topic_prompt", llm="gpt4", output="topic")
        # Generate question
        .generate_text(template="q_prompt", llm="gpt4", output="question")
        # Generate answer
        .generate_text(template="a_prompt", llm="gpt4", output="answer")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("question")
                .assistant("answer"),
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Use Conv() builder**: Recommended over string format for type safety
2. **Validate conversations**: Always use `.validate_conversation()`
3. **System messages**: Include clear system prompts
4. **Multi-turn**: Build complex dialogues with multiple exchanges
5. **Reasoning content**: Use `.think()` for chain-of-thought data
6. **Format-specific methods**: Use `.render_sft()`, `.render_dpo()`, `.render_grpo()` for specific training formats
7. **Diversity**: Use higher temperature (0.8-0.9) for diverse conversations
8. **Quality checks**: Add deduplication on conversation content

## Conv() Builder Methods

- `.system(content)` - Add system message
- `.user(content)` - Add user message
- `.assistant(content)` - Add assistant message
- `.tool(content)` - Add tool response message
- `.tool_calls(calls)` - Add tool calls (list or string)
- `.think(content)` - Add reasoning content

## Reference

For more examples, see:
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_steps.py` (lines 328-391)
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_judge.py`
