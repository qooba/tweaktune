"""
Validation and Quality Checks Example

Demonstrates various validation techniques:
- JSON schema validation
- Language detection
- Custom Python validators
- Data filtering

Based on test_steps.py (check_language, filter tests)
"""

from tweaktune import Pipeline
import json

def main():
    print("Example 1: Language detection")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"text": {{text|jstr}}, "lang": "english"}""")
        .iter_range(10)
            .add_column("text", lambda data: [
                "Hello, how are you?",
                "Bonjour, comment allez-vous?",
                "Hola, ¿cómo estás?",
                "Good morning everyone!",
                "Guten Morgen!",
                "Nice to meet you.",
                "Buongiorno!",
                "How's the weather today?",
                "¿Qué tal el tiempo hoy?",
                "What a beautiful day!"
            ][data["index"]])

            # Only keep English text
            .check_language(
                input="text",
                language="english",
                precision=0.9,
                detect_languages=["english", "french", "german", "spanish"]
            )

            .write_jsonl(path="11_english_only.jsonl", template="output")
        .run())
    print("Filtered to English only\n")

    print("Example 2: Custom validation")
    def validate_email(context):
        data = context["data"]
        email = data.get("email", "")

        # Check email format
        if "@" not in email or "." not in email:
            raise ValueError("Invalid email format")

        # Check domain
        if not email.endswith(".com"):
            raise ValueError("Email must end with .com")

        return True

    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"email": {{email|jstr}}, "name": {{name|jstr}}}""")
        .iter_range(8)
            .add_column("name", lambda data: f"User{data['index']}")
            .add_column("email", lambda data: [
                "user0@example.com",      # Valid
                "user1@test.net",          # Invalid (.net)
                "user2example.com",        # Invalid (no @)
                "user3@company.com",       # Valid
                "invalid@",                # Invalid
                "user5@service.com",       # Valid
                "@example.com",            # Invalid
                "user7@domain.com"         # Valid
            ][data["index"]])

            .validate(validate_email)

            .write_jsonl(path="11_valid_emails.jsonl", template="output")
        .run())
    print("Validated emails\n")

    print("Example 3: Data quality filters")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "text": {{text|jstr}},
  "length": {{length}},
  "word_count": {{word_count}}
}
""")
        .iter_range(10)
            .add_column("text", lambda data: [
                "Short",
                "This is a medium length text with several words in it.",
                "A",
                "Another example of text that has moderate length.",
                "Too short",
                "This text is long enough and contains many words that make it suitable for our dataset requirements.",
                "No",
                "Quality text needs to have sufficient length and word count.",
                "Okay",
                "Final example with adequate length and complexity."
            ][data["index"]])

            # Calculate metrics
            .add_column("length", lambda data: len(data["text"]))
            .add_column("word_count", lambda data: len(data["text"].split()))

            # Apply quality filters
            .filter(lambda data: data["length"] > 20)  # Minimum 20 characters
            .filter(lambda data: data["word_count"] >= 5)  # At least 5 words

            .write_jsonl(path="11_quality_text.jsonl", template="output")
        .run())
    print("Filtered by quality metrics\n")

    print("Example 4: Range validation")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "product": {{product|jstr}},
  "price": {{price}},
  "rating": {{rating}}
}
""")
        .iter_range(10)
            .add_random("price", 1, 200)
            .add_random("rating", 1, 5)
            .add_column("product", lambda data: f"Product_{data['index']}")

            # Filter by valid ranges
            .filter("price >= 10 and price <= 150")
            .filter("rating >= 3")  # Only 3+ star ratings

            .write_jsonl(path="11_valid_products.jsonl", template="output")
        .run())
    print("Validated product ranges\n")

    # Show results
    print("English-only texts:")
    with open("11_english_only.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            print(f"  - {data['text']}")

    print(f"\nValid emails:")
    with open("11_valid_emails.jsonl", "r") as f:
        count = 0
        for line in f:
            data = json.loads(line)
            count += 1
            print(f"  - {data['email']}")
        print(f"  Total: {count}/8 valid")

    print(f"\nQuality texts:")
    with open("11_quality_text.jsonl", "r") as f:
        count = 0
        for line in f:
            data = json.loads(line)
            count += 1
            print(f"  - {data['text'][:50]}... ({data['word_count']} words)")
        print(f"  Total: {count}/10 passed quality checks")

if __name__ == "__main__":
    main()
