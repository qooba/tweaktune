from pydantic import BaseModel, Field

#{
#  "scores": {
#    "intent_alignment": <int 1-5>,
#    "tool_choice_accuracy": <int 1-5>,
#    "argument_accuracy": <int 1-5>,
#    "response_quality": <int 1-5>,
#    "overall_coherence": <int 1-5>,
#    "safety": <int 1-5>
#  },
#  "rationales": {
#    "faithfulness": "<brief rationale>",
#    "clarity": "<brief rationale>",
#    "conciseness": "<brief rationale>",
#    "relevance": "<brief rationale>",
#    "creativity": "<brief rationale>"
#  }
#}

class JudgeResponse(BaseModel):
    intent_alignment: int = Field(..., ge=1, le=5, description="How well the response aligns with the user's intent.")
    tool_choice_accuracy: int = Field(..., ge=1, le=5, description="Accuracy of the chosen tool for the task.")
    argument_accuracy: int = Field(..., ge=1, le=5, description="Correctness of the arguments provided to the tool.")
    response_quality: int = Field(..., ge=1, le=5, description="Overall quality of the response.")
    overall_coherence: int = Field(..., ge=1, le=5, description="Coherence and logical flow of the response.")
    safety: int = Field(..., ge=1, le=5, description="Safety and appropriateness of the response.")
    faithfulness: str = Field(..., description="Rationale for faithfulness score.")
    clarity: str = Field(..., description="Rationale for clarity score.")
    conciseness: str = Field(..., description="Rationale for conciseness score.")
    relevance: str = Field(..., description="Rationale for relevance score.")
    creativity: str = Field(..., description="Rationale for creativity score.")

class JudgeLiteResponse(BaseModel):
    intent_alignment: int = Field(..., ge=1, le=5, description="How well the response aligns with the user's intent.")
    tool_choice_accuracy: int = Field(..., ge=1, le=5, description="Accuracy of the chosen tool for the task.")
    argument_accuracy: int = Field(..., ge=1, le=5, description="Correctness of the arguments provided to the tool.")
    response_quality: int = Field(..., ge=1, le=5, description="Overall quality of the response.")
    overall_coherence: int = Field(..., ge=1, le=5, description="Coherence and logical flow of the response.")
    safety: int = Field(..., ge=1, le=5, description="Safety and appropriateness of the response.")



def test_judge_response():
    print(JudgeResponse.model_json_schema())


def test_judge_lite_response():
    print(JudgeLiteResponse.model_json_schema())
