import importlib.resources

def __getattr__(name):
    """
    Dynamically import the module specified by `name` from the current package.
    """
    filename = f"{name}.j2"
    with importlib.resources.files(__package__).joinpath(filename).open("r", encoding="utf-8") as f:
        return f.read()
    

output: str
"""Output template for function calling."""

tool_call: str
"""Output template for function calling."""


output_raw: str
"""Output raw template for function calling."""

output_legacy: str
"""Output legacy template for function calling."""