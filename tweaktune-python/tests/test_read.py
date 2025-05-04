import pytest
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum
import tempfile
import shutil
import polars as pl


def test_mixed(request, output_dir, data_dir):
    """Test the basic functionality of the pipeline."""

    with open(f"{data_dir}/functions_micro.json", "w") as f:
        f.write("""[{"name": "function1", "description": "This is function 1."}, {"name": "function2", "description": "This is function 2."}]""")
    with open(f"{data_dir}/personas_micro.jsonl", "w") as f:   
        f.write("""{"name": "persona1", "description": "This is persona 1."}\n{"name": "persona2", "description": "This is persona 2."}""")

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_json_dataset("functions",f"{data_dir}/functions_micro.json")\
        .with_jsonl_dataset("personas",f"{data_dir}/personas_micro.jsonl")\
        .with_mixed_dataset("mixed",["functions", "personas"])\
        .with_template("output", """{"mixed": {{mixed|jstr}} }""")\
    .iter_dataset("mixed")\
        .write_jsonl(path=output_file, template="output")\
        .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "mixed" in item
    assert "functions" in item["mixed"]
    assert "personas" in item["mixed"]


csv_testdata = [
    ("functions_micro1.csv", ",", True,"""name,description\nfunction1,This is function 1.\nfunction2,This is function 2."""),
    ("functions_micro2.csv", ",", False,"""function1,This is function 1.\nfunction2,This is function 2."""),
    ("functions_micro3.csv", ";", True,"""name;description\nfunction1;This is function 1.\nfunction2;This is function 2."""),
    ("functions_micro4.csv", ";", False,"""function1;This is function 1.\nfunction2;This is function 2."""),
]
@pytest.mark.parametrize("file_name,delimeter,has_header,data", csv_testdata)
def test_csv1_read(request, output_dir, data_dir, file_name, delimeter, has_header, data):
    """Test the basic functionality of the pipeline."""

    with open(f"{data_dir}/{file_name}", "w") as f:
        f.write("""name,description\nfunction1,This is function 1.\nfunction2,This is function 2.""")

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_csv_dataset("functions",f"{data_dir}/{file_name}", delimiter=delimeter, has_header=has_header)\
        .with_template("output", """{"functions": {{functions|jstr}} }""")\
    .iter_dataset("functions")\
        .write_jsonl(path=output_file, template="output")\
        .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "functions" in item
    if has_header:
        assert "name" in item["functions"]
        assert "description" in item["functions"]


def test_parquet(request, output_dir, data_dir):
    """Test the basic functionality of the pipeline."""

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_parquet_dataset("functions","./tweaktune-python/tests/example.parquet")\
        .with_template("output", """{"functions": {{functions|jstr}} }""")\
    .iter_dataset("functions")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "functions" in item

def test_db(request, output_dir, data_dir):
    """Test the basic functionality of the pipeline."""

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_db_dataset("functions","sqlite://tweaktune-python/tests/test.db", "select * from `functions`")\
        .with_template("output", """{"functions": {{functions|jstr}} }""")\
    .iter_dataset("functions")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "functions" in item

def test_arrow(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""

    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_arrow_dataset("functions", arrow_dataset())\
        .with_template("output", """{"functions": {{functions|jstr}} }""")\
    .iter_dataset("functions")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "functions" in item
    assert len(lines) == 10





#def test_prepare_example_parquet(request, output_dir):
#    """Prepare an example parquet file using polars."""
#    df = pl.DataFrame({
#        "name": ["function1", "function2"],
#        "description": ["This is function 1.", "This is function 2."]
#    })
#    df.write_parquet("./example.parquet")

#def test_create_sqllite_database():
#    """Create a SQLite database and a table with some data."""
#    import sqlite3
#    import os
#
#    # Create a temporary SQLite database
#    db_file = "test.db"
#    conn = sqlite3.connect(db_file)
#    cursor = conn.cursor()
#
#    # Create a table
#    cursor.execute('''
#        CREATE TABLE functions (
#            id INTEGER PRIMARY KEY,
#            name TEXT,
#            description TEXT
#        )
#    ''')
#
#    # Insert some data
#    cursor.execute("INSERT INTO functions (name, description) VALUES ('function1', 'This is function 1.')")
#    cursor.execute("INSERT INTO functions (name, description) VALUES ('function2', 'This is function 2.')")
#
#    # Commit and close the connection
#    conn.commit()
#    conn.close()
#
#    return db_file