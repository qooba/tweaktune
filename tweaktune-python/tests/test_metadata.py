import json
import sqlite3
from tweaktune import Pipeline, Metadata

def test_metadata(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"


    metadata = Metadata(path=f"{output_dir}/.tweaktune", enabled=True)

    products = ["car", "bike", "scooter", "skateboard", "rollerblades"]

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", """{"hello": "{{value}}"}""")
    .iter_range(number)
        .log("info")
        .add_column("question", lambda data: f"Hello! What is the weather ! How are you ! I want to buy {products[data['index']]}? How much does it cost?")
        .add_column("call", lambda data: {"name": "test", "arguments": {"x": data["index"]}})
        .check_hash(input="call")
        .check_simhash(input="question", treshold=10)
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number

    conn = sqlite3.connect(f"{output_dir}/.tweaktune/state/state.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(tables)
    assert ('runs',) in tables
    assert ('items',) in tables
    assert ('hashes',) in tables
    assert ('simhashes',) in tables

    cursor.execute("SELECT * FROM runs;")
    runs = cursor.fetchall()
    print(runs)
    assert len(runs) == 1

    cursor.execute("SELECT * FROM items;")
    items = cursor.fetchall()
    print(items)
    assert len(items) == number

    cursor.execute("SELECT * FROM hashes;")
    hashes = cursor.fetchall()
    print(hashes)
    assert len(hashes) == number

    cursor.execute("SELECT * FROM simhashes;")
    simhashes = cursor.fetchall()
    print(simhashes)
    assert len(simhashes) == number

