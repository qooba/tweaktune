"""
Function Calling Pipeline Template

Generates function calling / tool use conversations.
"""

from pydantic import Field
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path


# ===== Define Tool Functions =====
def get_weather(
    location: str = Field(..., description="City name or location"),
    unit: str = Field("celsius", description="Temperature unit: celsius or fahrenheit")
):
    """Get current weather information for a specific location"""
    pass


def search_web(
    query: str = Field(..., description="Search query"),
    num_results: int = Field(5, description="Number of results to return", ge=1, le=10)
):
    """Search the web for information"""
    pass


def calculate(
    expression: str = Field(..., description="Mathematical expression to evaluate")
):
    """Calculate a mathematical expression"""
    pass


def send_email(
    to: str = Field(..., description="Recipient email address"),
    subject: str = Field(..., description="Email subject"),
    body: str = Field(..., description="Email body content")
):
    """Send an email to a recipient"""
    pass


def main():
    # ===== Configuration =====
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/function_calling.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Build and Run Pipeline =====
    (Pipeline(
        name="function-calling-pipeline",
        metadata=Metadata(description="Generate function calling conversations")
    )
        # Worker configuration
        .with_workers(4)

        # ===== Resource Configuration =====
        # Tools dataset: Load Python functions as tools
        .with_tools_dataset("available_tools", [
            get_weather,
            search_web,
            calculate,
            send_email
        ])

        # LLM: OpenAI GPT-4
        .with_llm_openai("gpt4", api_key, "gpt-4")

        # Templates: Define prompts for generation
        .with_template(
            "question_prompt",
            """Generate a realistic user question that would require using this tool: {{selected_tools[0].name}}

Tool description: {{selected_tools[0].description}}

Create a natural, specific question that clearly needs this tool to answer."""
        )
        .with_template(
            "args_prompt",
            """Generate appropriate JSON arguments for calling the tool {{selected_tools[0].name}} to answer this question:

Question: {{question}}

Tool schema: {{selected_tools[0]}}

Provide only the arguments as a valid JSON object."""
        )
        .with_template(
            "final_answer_prompt",
            """Based on the tool response, generate a natural final answer to the user's question.

Question: {{question}}
Tool response: {{tool_response}}

Provide a clear, helpful answer."""
        )

        # ===== Start Iteration =====
        .iter_range(100)  # Generate 100 function calling examples

        # ===== Pipeline Steps =====

        # Step 1: Sample tools for this conversation
        .sample_tools("available_tools", size=1, output="selected_tools")

        # Step 2: Set system message
        .add_column(
            "system",
            lambda data: "You are a helpful assistant with access to tools. Use the appropriate tools to answer user questions accurately."
        )

        # Step 3: Generate user question
        .generate_text(
            template="question_prompt",
            llm="gpt4",
            output="question",
            max_tokens=100,
            temperature=0.8
        )

        # Step 4: Generate tool call arguments
        .generate_json(
            template="args_prompt",
            llm="gpt4",
            output="tool_args",
            json_path="$"
        )

        # Step 5: Render the tool call in proper format
        .render_tool_call(
            tool="selected_tools[0].name",
            arguments="tool_args",
            output="tool_call"
        )

        # Step 6: Simulate tool response (mock data)
        # In production, you might call actual APIs here
        .add_column(
            "tool_response",
            lambda data: '{"result": "Simulated tool response based on the query"}'
        )

        # Step 7: Generate final answer based on tool response
        .generate_text(
            template="final_answer_prompt",
            llm="gpt4",
            output="final_answer",
            max_tokens=256,
            temperature=0.7
        )

        # Step 8: Build conversation with tool calls
        .render_conversation(
            conversation=Conv()
                .system("system")            # System message
                .user("question")            # User question
                .tool_calls(["tool_call"])   # Assistant tool call
                .tool("tool_response")       # Tool response
                .assistant("final_answer"),  # Final answer
            tools="selected_tools",  # Include tool definitions
            output="conversation"
        )

        # Step 9: Validate tools and conversation format
        .validate_tools("selected_tools")
        .validate_conversation("conversation")

        # ===== Output =====
        .write_jsonl(
            path=str(output_path),
            value="conversation"  # Write conversation object
        )

        # ===== Execute =====
        .run()
    )

    print(f"✓ Function calling generation completed!")
    print(f"✓ Generated 100 tool use conversations")
    print(f"✓ Output written to: {output_path}")
    print(f"\nOutput format: OpenAI function calling format with tools")


if __name__ == "__main__":
    main()
