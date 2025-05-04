import tempfile
import shutil
import pytest
import os
import polars as pl


@pytest.fixture(scope="session")
def data_dir():
    """Fixture to create a temporary directory for testing."""
    data_dir = tempfile.mkdtemp()

    yield data_dir

    shutil.rmtree(data_dir)

@pytest.fixture(scope="session")
def output_dir():
    """Fixture to create a temporary directory for testing."""
    output_dir = tempfile.mkdtemp()

    yield output_dir

    shutil.rmtree(output_dir)

@pytest.fixture(scope="session")
def arrow_dataset():
    """Fixture to create a temporary Arrow dataset for testing."""
    return arrow_builder

def arrow_builder():
    arrow_dataset = pl.DataFrame({
        "name": [
            "Apple", "Banana", "Carrot", "Doughnut", "Eggplant",
            "Fish Fillet", "Grapes", "Honey", "Ice Cream", "Juice"
        ],
        "description": [
            "Fresh red apple.", "Ripe yellow banana.", "Organic orange carrot.",
            "Glazed sweet doughnut.", "Purple eggplant, firm.",
            "Boneless white fish fillet.", "Seedless green grapes.",
            "Pure natural honey.", "Vanilla flavored ice cream.", "Fresh orange juice."
        ],
        "price": [0.5, 0.3, 0.2, 1.0, 1.5, 3.0, 2.0, 4.5, 2.5, 1.2],
        "size": ["medium", "large", "small", "medium", "large", "medium", "small", "small", "large", "medium"],
        "weight": [150, 120, 80, 60, 300, 200, 100, 250, 500, 330],
        "category": [
            "Fruit", "Fruit", "Vegetable", "Bakery", "Vegetable",
            "Seafood", "Fruit", "Condiment", "Dessert", "Beverage"
        ]
    })

    return arrow_dataset.to_arrow().to_reader()



