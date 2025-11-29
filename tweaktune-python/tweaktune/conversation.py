from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Message:
    role: str
    content: str
    type: Optional[str] = None


class ConversationBuilder:
    def __init__(self):
        self.messages: List[Message] = []

    def system(self, content: str):
        self.messages.append(Message(role="system", content=content))
        return self

    def user(self, content: str):
        self.messages.append(Message(role="user", content=content))
        return self

    def assistant(self, content: str):
        self.messages.append(Message(role="assistant", content=content))
        return self

    def tool_calls(self, calls: Union[str, List[str]]):
        if isinstance(calls, str):
            calls = [calls]

        content = ",".join(calls)
        content = f"[{content}]"
        self.messages.append(Message(role="assistant", content=content, type="tool_calls"))
        return self

    def tool(self, content: str):
        self.messages.append(Message(role="tool", content=content))
        return self

    def think(self, content: str):
        self.messages.append(Message(role="assistant", content=content, type="think"))
        return self

    def build(self):
        parts = []
        for msg in self.messages:
            if msg.type:
                parts.append(f"@{msg.role}:{msg.type}({msg.content})")
            else:
                parts.append(f"@{msg.role}:{msg.content}")
        return "|".join(parts)
