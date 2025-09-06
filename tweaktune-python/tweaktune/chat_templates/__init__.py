from tweaktune.tweaktune import EmbedChatTemplates

def __getattr__(name):
    """
    Dynamically import the module specified by `name` from the current package.
    """
    if name == "bielik":
        return EmbedChatTemplates.Bielik.template()
    raise AttributeError(f"module {__name__} has no attribute {name}")

bielik: str
"""Output template for function calling."""