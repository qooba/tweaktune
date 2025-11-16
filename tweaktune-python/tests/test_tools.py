import json
from typing import Optional

from pydantic import Field

from tweaktune import Pipeline


def place_order(
    user_id: str = Field(..., description="ID użytkownika składającego zamówienie"),
    product_id: str = Field(..., description="ID zamawianego produktu"),
    quantity: int = Field(..., description="Ilość zamawianego produktu"),
    delivery_address: str = Field(..., description="Adres dostawy"),
):
    """Złóż zamówienie na produkt w e-bazarze."""
    pass


def get_order_status(
    order_id: str = Field(..., description="ID zamówienia"),
):
    """Pobierz status zamówienia w e-bazarze."""
    pass


def search_products(
    query: str = Field(..., description="Fraza wyszukiwania produktów"),
    category: Optional[str] = Field(None, description="Kategoria produktów"),
    min_price: Optional[float] = Field(None, description="Minimalna cena"),
    max_price: Optional[float] = Field(None, description="Maksymalna cena"),
):
    """Wyszukaj produkty w e-bazarze według frazy, kategorii i zakresu cen."""
    pass


def list_categories():
    """Wyświetl dostępne kategorie produktów w e-bazarze."""
    pass


def add_review(
    product_id: str = Field(..., description="ID produktu do recenzji"),
    user_id: str = Field(..., description="ID użytkownika dodającego recenzję"),
    rating: int = Field(..., ge=1, le=5, description="Ocena produktu (1-5)"),
    comment: Optional[str] = Field(None, description="Komentarz do recenzji"),
):
    """Dodaj recenzję produktu w e-bazarze."""
    pass


def get_product_details(
    product_id: str = Field(..., description="ID produktu"),
):
    """Pobierz szczegóły produktu z e-bazaru."""
    pass


def list_user_orders(
    user_id: str = Field(..., description="ID użytkownika"),
):
    """Wyświetl listę zamówień użytkownika w e-bazarze."""
    pass


def test_tools_sample(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    OUTPUT_TEMPLATE = """{"function": {{function[0]}}, "all_functions": {{all_functions}} }"""

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_tools_dataset(
            "functions",
            [
                place_order,
                get_order_status,
                search_products,
                list_categories,
                add_review,
                get_product_details,
                list_user_orders,
            ],
        )
        .with_template("output", OUTPUT_TEMPLATE)
        .iter_range(10)
        .sample("functions", 1, "function")
        .sample("functions", 2, "all_functions")
        .validate_tools("function")
        .validate_tools("all_functions")
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    item = json.loads(lines[0])
    assert "function" in item
    assert "all_functions" in item
    assert len(lines) == 10


def test_tools_sample(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    OUTPUT_TEMPLATE = """{"all_functions": {{all_functions|tojson}} }"""

    #   , search_products, list_categories, , ,
    # .with_tools_dataset("functions", [place_order, get_order_status, search_products, list_categories, add_review, get_product_details, list_user_orders])\

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_tools_dataset("functions", [search_products, list_categories])
        .with_template("output", OUTPUT_TEMPLATE)
        .iter_range(10)
        .sample("functions", 2, "all_functions")
        .validate_tools("all_functions")
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    item = json.loads(lines[0])
    assert "all_functions" in item
    assert len(lines) == 10


def test_tools_sample_2(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    OUTPUT_TEMPLATE = """{"all_functions": {{all_functions|tojson}} }"""

    #   , search_products, list_categories, , ,
    # .with_tools_dataset("functions", [place_order, get_order_status, search_products, list_categories, add_review, get_product_details, list_user_orders])\

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_tools_dataset("functions", [search_products, list_categories])
        .with_template("output", OUTPUT_TEMPLATE)
        .iter_range(10)
        .sample_tools("functions", 2, "all_functions")
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    item = json.loads(lines[0])
    assert "all_functions" in item
    assert len(lines) == 10
