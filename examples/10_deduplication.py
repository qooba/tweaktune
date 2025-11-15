"""
Deduplication Example

Demonstrates various deduplication methods:
- Exact hash deduplication
- Fuzzy simhash deduplication
- Semantic embedding deduplication

Based on test_metadata.py
"""

from tweaktune import Pipeline, Metadata
import sqlite3
import json

def main():
    # Enable metadata for deduplication
    metadata = Metadata(path="./.tweaktune_example", enabled=True)

    products = [
        "laptop computer",
        "laptop",  # Similar to first
        "smartphone device",
        "smartphone",  # Similar to third
        "laptop computer",  # Exact duplicate of first
        "tablet device",
        "smart phone",  # Very similar to "smartphone"
        "portable computer",  # Semantically similar to "laptop"
        "headphones",
        "wireless headphones"  # Similar to previous
    ]

    print("Running deduplication pipeline...")
    (Pipeline(name="dedup_example", metadata=metadata)
        .with_workers(1)

        # Configure embedding model for semantic deduplication
        .with_embeddings_e5(
            name="e5-small",
            model_repo="intfloat/e5-small"
        )

        .with_template("output", """{"product": {{product|jstr}}, "index": {{index}}}""")

        .iter_range(len(products))
            .add_column("product", lambda data: products[data["index"]])

            # Apply all three deduplication methods
            # Items that match will be marked as failed and won't be written

            # 1. Exact hash - catches identical strings
            .check_hash(input="product")

            # 2. SimHash - catches very similar strings (typos, minor variations)
            .check_simhash(input="product", threshold=3)

            # 3. Embeddings - catches semantic duplicates
            .check_embedding(
                input="product",
                embedding="e5-small",
                threshold=0.05,
                similarity_output="similarity"
            )

            .write_jsonl(path="10_unique_products.jsonl", template="output")
        .run())

    print("Deduplication complete\n")

    # Analyze results
    print("Results:")
    print(f"  Original items: {len(products)}")

    with open("10_unique_products.jsonl", "r") as f:
        unique_items = f.readlines()
        print(f"  Unique items: {len(unique_items)}")
        print(f"  Duplicates removed: {len(products) - len(unique_items)}")

    print("\nUnique products:")
    with open("10_unique_products.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            print(f"  - {data['product']}")

    # Query metadata database
    print("\nMetadata statistics:")
    conn = sqlite3.connect(".tweaktune_example/state/state.db")

    total_hashes = conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
    total_simhashes = conn.execute("SELECT COUNT(*) FROM simhashes").fetchone()[0]
    total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    print(f"  Stored hashes: {total_hashes}")
    print(f"  Stored simhashes: {total_simhashes}")
    print(f"  Stored embeddings: {total_embeddings}")

    # Get runs info
    runs = conn.execute("SELECT pipeline_name, total_items FROM runs").fetchall()
    for run in runs:
        print(f"  Pipeline '{run[0]}' processed {run[1]} items")

    conn.close()

    # Cleanup
    import shutil
    shutil.rmtree(".tweaktune_example")
    print("\nCleaned up metadata directory")

if __name__ == "__main__":
    main()
