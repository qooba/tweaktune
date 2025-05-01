import pytest
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum
import tempfile
import shutil


def test_basic(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_template("output", """{"hello": "world"}""")\
    .iter_range(number)\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number

def test_dicts(request, output_dir):
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

def test_tools(request, data_dir, output_dir):
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

def test_pydantic(request, output_dir):
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


def test_openapi(request, output_dir):
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