import json
from datasets import Dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template

class BaseDataset:

    def with_tokenizer(self, model_name: str, chat_template: str, eos_token: str, mapping: dict = None):
        """
        Prepare the tokenizer for the model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if mapping is None:
            mapping = {
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt"
            }

        tokenizer = get_chat_template(
            tokenizer,
            chat_template = (chat_template, eos_token,),
            mapping = mapping,
            map_eos_token = True,
        )

        self.tokenizer = tokenizer
        return self

    def _normalize(self, path) -> list:
        """
        Normalize the dataset from a given path.
        This method should be overridden by subclasses to implement specific normalization logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def build(self, path) -> Dataset:

        if not hasattr(self, "tokenizer"):
            raise ValueError("Tokenizer is not set. Please call 'with_tokenizer' before building the dataset.")

        dataset = self._normalize(path)

        train_dataset = []

        for d in dataset:
            messages= d["conversation"]
            tools = d["tools"]

            inputs = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True, return_tensors = "pt", enable_thinking=False, tools=tools)
            train_dataset.append({"text": inputs})

        return Dataset.from_list(train_dataset)


class ToolsDataset(BaseDataset):
    def _normalize(self, path: str = None) -> list:
        dataset = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    data = json.loads(line)
                    data["tools"] = []
                    for f in data['function_descriptions']:
                        f["parameters"]["properties"] = json.loads(f["parameters"]["properties"])
                        del f["strict"]
                        del f["type"]
                        if not f["parameters"]["required"]:
                            f["parameters"]["required"] = []

                        tool = {
                            "type": "function",
                            "function": f
                        }

                        data["tools"].append(tool)
                    del data['function_descriptions']

                    for c in data["conversation"]:
                        if c["speaker"] == "human":
                            c["role"] = "user"
                            c["content"] = c["message"]
                            del c["message"]
                            del c["speaker"]
                        elif c.get("action") == "function-call":
                            c["role"] = "assistant"
                            c["details"]["arguments"] = json.loads(c["details"]["arguments"])
                            c["content"]=f"<tool_call>{json.dumps(c['details'], ensure_ascii=False)}</tool_call>"
                            del c["action"]
                            del c["details"]
                            del c["speaker"]
                        elif c.get("speaker") == "assistant":
                            c["role"] = "assistant"
                            c["content"] = c["message"]
                            del c["message"]
                            del c["speaker"]

                    dataset.append(data)
                except Exception as ex:
                    pass

        return dataset
