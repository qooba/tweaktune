# TweakTune Synthesizer Plugin for Claude Code

Interactive Claude Code plugin for designing and generating TweakTune pipelines for synthetic dataset creation.

## Overview

The TweakTune Synthesizer plugin provides an intelligent assistant that guides you through creating production-ready data synthesis pipelines using the TweakTune library. Through an interactive Q&A process, it generates complete, runnable code for:

- **Text Generation** - Articles, summaries, creative writing
- **JSON/Structured Data** - Personas, entities, labeled datasets
- **Conversations** - Multi-turn dialogues for chat fine-tuning
- **Function Calling** - Tool use examples for agent training

## Features

### Interactive Q&A Flow

The skill guides you through 7 phases of configuration:

1. **Task Discovery** - What type of data to synthesize
2. **Data Sources** - Where seed data comes from
3. **LLM Configuration** - Which provider and model to use
4. **Template Design** - How to structure prompts
5. **Quality Checks** - Deduplication and validation
6. **Output Configuration** - Where to save results
7. **Code Generation** - Complete working pipeline

### What You Get

For each pipeline, the skill generates:

- ✅ Complete Python pipeline script with comments
- ✅ Pydantic models for structured data (if needed)
- ✅ Jinja2 templates for complex prompts (optional)
- ✅ Quality checks (deduplication, language detection)
- ✅ Validation steps (JSON schema, conversation format)
- ✅ Error handling and best practices
- ✅ requirements.txt with dependencies
- ✅ README with usage instructions

### Included Resources

**Example Documentation:**
- `examples/text-generation.md` - Text synthesis patterns
- `examples/json-generation.md` - Structured data with Pydantic
- `examples/conversations.md` - Multi-turn dialogue synthesis
- `examples/function-calling.md` - Tool use dataset generation

**Code Templates:**
- `templates/basic-pipeline.py` - Minimal pipeline scaffold
- `templates/text-gen-pipeline.py` - Complete text generation
- `templates/json-gen-pipeline.py` - Structured JSON generation
- `templates/conversation-pipeline.py` - Conversation synthesis
- `templates/function-call-pipeline.py` - Function calling examples

## Installation

### Prerequisites

- [Claude Code](https://code.claude.com) installed
- TweakTune library installed: `pip install tweaktune`

### Install the Plugin

```bash
# In Claude Code, add the TweakTune marketplace
/plugin marketplace add qooba/tweaktune

# Install the synthesizer plugin
/plugin install tweaktune-synthesizer@tweaktune-plugins
```

### Verify Installation

```bash
# List installed plugins
/plugin list

# You should see: tweaktune-synthesizer
```

## Usage

### Quick Start

Simply ask Claude Code about synthesizing data:

```
I want to create a dataset for fine-tuning with conversation data.
```

The skill will automatically activate and guide you through the setup.

### Example Interactions

**Text Generation:**
```
Help me generate articles from topics for training a summarization model.
```

**JSON Data:**
```
I need to create synthetic personas in JSON format with Pydantic validation.
```

**Conversations:**
```
Generate multi-turn conversations for chat fine-tuning.
```

**Function Calling:**
```
Create a dataset of function calling examples using my Python functions.
```

### What Happens

1. The skill asks about your requirements
2. You answer questions about data sources, LLM, output format
3. The skill generates complete pipeline code
4. You get a ready-to-run Python script with all setup

### Example Output

After answering questions, you'll get files like:

```
output/
├── pipeline.py              # Main pipeline script
├── requirements.txt         # Dependencies
├── templates/
│   └── prompt.j2           # Jinja2 templates (if needed)
└── README.md               # Usage instructions
```

## Advanced Features

### Custom Validation

The skill can add custom validation logic:

```python
.validate(lambda data: your_validation_function(data))
```

### Quality Checks

- **Deduplication**: Hash-based, fuzzy (simhash), or semantic (embeddings)
- **Language Detection**: Filter by language with confidence threshold
- **Schema Validation**: JSON schema validation for structured data
- **Format Validation**: Conversation and tool calling format checks

### Multiple Data Sources

Supports various input formats:

- Parquet, CSV, JSONL, JSON files
- HuggingFace datasets
- Databases (via ConnectorX)
- OpenAPI specifications
- Python functions and Pydantic models
- Or generate from scratch with `.iter_range()`

### LLM Providers

- OpenAI API
- Azure OpenAI
- Generic API (Ollama, vLLM, etc.)
- Local models (Unsloth, MistralRS)

## Best Practices

The skill automatically includes:

- ✓ API keys from environment variables (never hardcoded)
- ✓ Output directory creation
- ✓ Error handling for missing configuration
- ✓ Meaningful pipeline names for debugging
- ✓ Metadata tracking
- ✓ Worker configuration based on API limits
- ✓ Comprehensive comments explaining each step

## Examples from the Skill

### Text Generation Pipeline

```python
from tweaktune import Pipeline, Metadata
import os

(Pipeline(name="text-generation")
    .with_workers(4)
    .with_jsonl_dataset("topics", "topics.jsonl")
    .with_llm_openai("gpt4", os.getenv("OPENAI_API_KEY"), "gpt-4")
    .with_template("prompt", "Generate article about: {{topic}}")
    .iter_dataset("topics")
    .generate_text(template="prompt", llm="gpt4", output="article")
    .check_hash("article")
    .write_jsonl(path="output.jsonl", template='{"article": "{{article}}"}')
    .run()
)
```

### Conversation Pipeline

```python
from tweaktune import Pipeline, Conv

(Pipeline(name="conversations")
    .with_llm_openai("gpt4", api_key, "gpt-4")
    .iter_range(100)
    .add_column("system", lambda d: "You are a helpful assistant.")
    .generate_text(template="Generate a question", llm="gpt4", output="question")
    .generate_text(template="Answer: {{question}}", llm="gpt4", output="answer")
    .render_conversation(
        conversation=Conv()
            .system("system")
            .user("question")
            .assistant("answer"),
        output="conversation"
    )
    .validate_conversation("conversation")
    .write_jsonl(path="output.jsonl", value="conversation")
    .run()
)
```

## Support

- **Documentation**: [TweakTune Docs](https://github.com/qooba/tweaktune)
- **Issues**: [GitHub Issues](https://github.com/qooba/tweaktune/issues)
- **Examples**: See `skills/tweaktune-synthesizer/examples/`

## Contributing

Contributions to improve the skill are welcome! The skill is located at:

```
.claude/skills/tweaktune-synthesizer/
tweaktune-plugin/skills/tweaktune-synthesizer/
```

## License

MIT License - see the main TweakTune repository for details.

## Learn More

- [TweakTune GitHub](https://github.com/qooba/tweaktune)
- [Claude Code Documentation](https://code.claude.com/docs)
- [Claude Code Skills Guide](https://docs.claude.com/en/docs/agents-and-tools/agent-skills)

---

**Made with ❤️ by the TweakTune Team**
