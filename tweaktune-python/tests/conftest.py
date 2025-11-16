import shutil
import tempfile

import polars as pl
import pytest
from tweaktune import Metadata


@pytest.fixture(scope="session")
def data_dir():
    """Fixture to create a temporary directory for testing."""
    data_dir = tempfile.mkdtemp()

    yield data_dir

    shutil.rmtree(data_dir)


@pytest.fixture(scope="session")
def metadata():
    """Fixture to set metadata."""
    yield Metadata(path="", enabled=False)


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
    arrow_dataset = pl.DataFrame(
        {
            "name": [
                "Apple",
                "Banana",
                "Carrot",
                "Doughnut",
                "Eggplant",
                "Fish Fillet",
                "Grapes",
                "Honey",
                "Ice Cream",
                "Juice",
            ],
            "description": [
                "Fresh red apple.",
                "Ripe yellow banana.",
                "Organic orange carrot.",
                "Glazed sweet doughnut.",
                "Purple eggplant, firm.",
                "Boneless white fish fillet.",
                "Seedless green grapes.",
                "Pure natural honey.",
                "Vanilla flavored ice cream.",
                "Fresh orange juice.",
            ],
            "price": [0.5, 0.3, 0.2, 1.0, 1.5, 3.0, 2.0, 4.5, 2.5, 1.2],
            "size": [
                "medium",
                "large",
                "small",
                "medium",
                "large",
                "medium",
                "small",
                "small",
                "large",
                "medium",
            ],
            "weight": [150, 120, 80, 60, 300, 200, 100, 250, 500, 330],
            "category": [
                "Fruit",
                "Fruit",
                "Vegetable",
                "Bakery",
                "Vegetable",
                "Seafood",
                "Fruit",
                "Condiment",
                "Dessert",
                "Beverage",
            ],
        }
    )

    return arrow_dataset.to_arrow().to_reader()


@pytest.fixture(scope="session")
def sqlite_database():
    """Create a SQLite database and a table with some data."""
    import sqlite3

    output_db_dir = tempfile.mkdtemp()
    # Create a temporary SQLite database
    db_file = f"{output_db_dir}/test.db"

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create a table
    cursor.execute(
        """
        CREATE TABLE functions (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT
        )
    """
    )

    # Insert some data
    cursor.execute(
        "INSERT INTO functions (name, description) VALUES ('function1', 'This is function 1.')"
    )
    cursor.execute(
        "INSERT INTO functions (name, description) VALUES ('function2', 'This is function 2.')"
    )

    # Commit and close the connection
    conn.commit()
    conn.close()

    yield db_file

    shutil.rmtree(output_db_dir)


@pytest.fixture(scope="session")
def parquet_file():
    """Prepare an example parquet file using polars."""

    output_pq_dir = tempfile.mkdtemp()
    pq_file = f"{output_pq_dir}/example.parquet"

    # Create a DataFrame with 10 shop items
    df = pl.DataFrame(
        {
            "name": [
                "Apple",
                "Banana",
                "Carrot",
                "Doughnut",
                "Eggplant",
                "Fish Fillet",
                "Grapes",
                "Honey",
                "Ice Cream",
                "Juice",
            ],
            "description": [
                "Fresh red apple.",
                "Ripe yellow banana.",
                "Organic orange carrot.",
                "Glazed sweet doughnut.",
                "Purple eggplant, firm.",
                "Boneless white fish fillet.",
                "Seedless green grapes.",
                "Pure natural honey.",
                "Vanilla flavored ice cream.",
                "Fresh orange juice.",
            ],
            "price": [0.5, 0.3, 0.2, 1.0, 1.5, 3.0, 2.0, 4.5, 2.5, 1.2],
            "size": [
                "medium",
                "large",
                "small",
                "medium",
                "large",
                "medium",
                "small",
                "small",
                "large",
                "medium",
            ],
        }
    )
    df.write_parquet(pq_file)

    yield pq_file
    shutil.rmtree(output_pq_dir)


@pytest.fixture(scope="session")
def j2_file():
    """Prepare an example j2 template file."""

    output_j2_dir = tempfile.mkdtemp()
    j2_file = f"{output_j2_dir}/example.j2"
    with open(j2_file, "w") as f:
        f.write("""{"hello": "{{value}}"}""")

    yield j2_file
    shutil.rmtree(output_j2_dir)


@pytest.fixture(scope="session")
def j2_dir():
    """Prepare an example j2 template directory."""

    output_j2_dir = tempfile.mkdtemp()
    j2_file = f"{output_j2_dir}/example.j2"
    with open(j2_file, "w") as f:
        f.write("""{"hello": "{{value}}"}""")

    yield output_j2_dir
    shutil.rmtree(output_j2_dir)


@pytest.fixture(scope="session")
def j2_file_yaml():
    """Prepare an example j2 template file."""

    output_j2_dir = tempfile.mkdtemp()
    j2_file = f"{output_j2_dir}/example_templates.yaml"
    with open(j2_file, "w") as f:
        f.write(
            """templates:
  input: >
    {"hello": "{{value}}"}
  output: >
    {"hello": "{{value}}"}
"""
        )

    yield j2_file
    shutil.rmtree(output_j2_dir)
