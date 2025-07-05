import json
from typing import Optional
from pydantic import BaseModel, Field

class JudgeRatings(BaseModel):
    """
    Judge ratings for the generated text.
    """
    accuracy: float = Field(description="Accuracy values (0-3): factual correctness of the text")
    relevance: float = Field(description="Relevance values (0-2): relevance of the text to the prompt")
    clarity: float = Field(description="Clarity values (0-2): clarity of the language used in the text")
    usefulness: float = Field(description="Usefulness values (0-3): usefulness of the text to the user")    

class PyStepWrapper:
    def __init__(self, step):
        self.step = step

    def process(self, context):
        context = json.loads(context)
        return json.dumps(self.step.process(context))
    
class PyConditionWrapper:
    def __init__(self, step):
        self.step = step

    def check(self, context):
        context = json.loads(context)
        return self.step.check(context["data"])

class LLMWrapper:

    def prepare_messages(self, messages: dict, json_schema: Optional[str]) -> Optional[str]:
        if json_schema:
            json_schema = json.loads(json_schema).get("schema", None)
            if json_schema:
                properties = json_schema.get("properties", {})
                if len(properties.keys()) > 0:
                    json_schema = json.dumps(json_schema, ensure_ascii=False)
                    system_message = [{"role": "system", "content": f"Always respond in valid JSON format, strictly following the provided schema. Do not include any extra text, explanations, or comments outside the JSON structure. Ensure that all responses adhere precisely to the given schema.\n\nThe expected schema is:\n\n{json_schema}"}]
                    messages = system_message + messages

        return messages
    
    def process(self, messages: dict, json_schema: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.1):
        raise NotImplementedError("This method should be implemented by subclasses.")

class UnslothWrapper(LLMWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def process(self, messages: dict, json_schema: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.1):
        messages = self.prepare_messages(messages, json_schema)
        inputs = self.tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
        output_ids = self.model.generate(input_ids = inputs, max_new_tokens = max_tokens, temperature = temperature, use_cache = True)
        input_size = inputs[0].shape[0]
        output_text = self.tokenizer.decode(output_ids[0][input_size:], skip_special_tokens=False)
        return output_text
    
class MistralrsWrapper(LLMWrapper):
    def __init__(self, runner):
        self.runner = runner

    def process(self, messages: dict, json_schema: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.1):
        from mistralrs import ChatCompletionRequest, Runner, Which
        messages = self.prepare_messages(messages, json_schema)

        req = ChatCompletionRequest(
                model="mistral",
                messages=messages,
                max_tokens=max_tokens,
                presence_penalty=1.0,
                temperature=temperature,
        )

#        if json_schema:
#            req = ChatCompletionRequest(
#                model="mistral",
#                messages=messages,
#                max_tokens=1024,
#                presence_penalty=1.0,
#                temperature=0.1,
#                grammar_type = "json_schema",
#                grammar = json.dumps(json.loads(json_schema)["schema"])
#            )


        res = self.runner.send_chat_completion_request(req)
        return res.choices[0].message.content
    
class PyStepValidatorWrapper:
    def __init__(self, func):
        self.func = func

    def process(self, context):
        context = json.loads(context)
        return self.func(context)

