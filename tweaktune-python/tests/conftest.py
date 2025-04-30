import tempfile
import shutil
import pytest
import os


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
