import pytest
import json
from tweaktune import Pipeline
import polars as pl
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum


def test_read_mixed(request, output_dir, data_dir):
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
def test_read_csv(request, output_dir, data_dir, file_name, delimeter, has_header, data):
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


def test_read_parquet(request, output_dir, data_dir, parquet_file):
    """Test the basic functionality of the pipeline."""

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_parquet_dataset("functions",parquet_file)\
        .with_template("output", """{"functions": {{functions|jstr}} }""")\
    .iter_dataset("functions")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "functions" in item
    assert "name" in item["functions"]
    assert "description" in item["functions"]

def test_read_db(request, output_dir, data_dir, sqlite_database):
    """Test the basic functionality of the pipeline."""

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_db_dataset("functions",f"sqlite://{sqlite_database}", "select * from `functions`")\
        .with_template("output", """{"functions": {{functions|jstr}} }""")\
    .iter_dataset("functions")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "functions" in item

def test_read_arrow(request, output_dir, data_dir, arrow_dataset):
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


def test_read_dicts(request, output_dir):
    """Test the dicts dataset functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    classes_list = [
        "Dog",
        "Cat",
        "Fish",
        "Bird",
        "Lizard",
    ]

    classes = [
        {
            "name": cls,
            "description": f"This is a {cls.lower()}",
            "type": "animal",
            "class": cls,
        }
        for cls in classes_list
    ]

    Pipeline()\
        .with_workers(1)\
        .with_dicts_dataset("json_list", classes)\
        .with_template("output", """{"class": {{json_object[0].class|jstr}} }""")\
    .iter_range(number)\
        .sample(dataset="json_list", size=1, output="json_object")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    for line in lines:
        data = json.loads(line)
        assert data["class"] in classes_list
    
    assert len(lines) == number

def test_read_tools(request, data_dir, output_dir):
    """Test the tools dataset functionality of the pipeline."""
    class ItemType(str, Enum):
        """
        Enum for item types.
        """
        electronics = "electronics"
        clothing = "clothing"
        food = "food"


    def my_function(
        id: int = Field(..., description="The ID of the item"),
        name: str = Field(..., description="The name of the item"),
        description: str = Field(..., description="The description of the item"),
        price: Optional[float] = Field(..., description="The price of the item"),
        quantity: int = Field(..., description="The quantity of the item"),
        type: ItemType = Field(..., description="The type of the item"),
    ):
        """This is a sample function."""
        pass

    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_tools_dataset("tools", [my_function])\
        .with_template("output", """{"tool": {{tool|jstr}} }""")\
    .iter_range(number)\
        .sample(dataset="tools", size=1, output="tool")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number

def test_read_pydantic(request, output_dir):
    """Test the pydantic dataset functionality of the pipeline."""
    class ItemType(str, Enum):
        """
        Enum for item types.
        """
        electronics = "electronics"
        clothing = "clothing"
        food = "food"


    class Item(BaseModel):
        """
        Item model for returning items.
        """
        id: int = Field(..., description="The ID of the item")
        name: str = Field(..., description="The name of the item")
        description: str = Field(..., description="The description of the item")
        price: Optional[float] = Field(..., description="The price of the item")
        quantity: int = Field(..., description="The quantity of the item")
        type: ItemType = Field(..., description="The type of the item")


    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_pydantic_models_dataset("pydantic_models", [Item])\
        .with_template("output", """{"pydantic_models": {{pydantic_model|jstr}} }""")\
    .iter_range(number)\
        .sample(dataset="pydantic_models", size=1, output="pydantic_model")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number


def test_read_openapi(request, output_dir):
    """Test the openapi dataset functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_openapi_dataset("openapi", "./tweaktune-python/tests/openapi.json")\
        .with_template("output", """{"api": {{openapi|jstr}} }""")\
    .iter_range(number)\
        .sample(dataset="openapi", size=1, output="openapi")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number


