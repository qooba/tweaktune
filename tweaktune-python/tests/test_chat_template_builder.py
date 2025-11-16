import json
from typing import List, Optional

from pydantic import BaseModel, Field

from tweaktune import ChatTemplateBuilder
from tweaktune.chat_templates import bielik


def search_products(
    query: str = Field(..., description="Fraza wyszukiwania produktów"),
    store: str = Field(..., description="Nazwa marki"),
    min_price: Optional[float] = Field(None, description="Minimalna cena"),
    max_price: Optional[float] = Field(None, description="Maksymalna cena"),
):
    """Wyszukaj produkty w wybranym sklepie według frazy i zakresu cen."""
    pass


def test_chat_template_builder(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""

    messages = [
        {
            "role": "user",
            "content": "Ile wyniesie indywidualna kwota napiwku dla każdej osoby w naszej grupie archeologów, jeśli całkowity rachunek za kolację wynosi 250 złotych, a w grupie jest nas pięcioro?",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculate_tip_split",
                        "arguments": {"total_bill": 250, "number_of_people": 5},
                    }
                },
                {
                    "function": {
                        "name": "calculate_tip_split",
                        "arguments": {"total_bill": 250, "number_of_people": 5},
                    }
                },
            ],
        },
        {"role": "tool", "content": '{"calculate_tip_split": {"individual_tip_amount": 50}}'},
        {"role": "tool", "content": '{"calculate_tip_split": {"individual_tip_amount": 50}}'},
        {
            "role": "assistant",
            "content": "Indywidualna kwota napiwku dla każdej osoby w waszej grupie archeologów wyniesie 50 złotych.",
        },
    ]

    chat_template = ChatTemplateBuilder(template=bielik).with_tools([search_products]).build()
    res = chat_template.render(messages=messages)
    assert (
        '<tool_call>{"name": "calculate_tip_split", "arguments": {"number_of_people":5,"total_bill":250}}</tool_call>'
        in res
    )
