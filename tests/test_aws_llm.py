from pydantic import BaseModel

from career_agent import aws_llm


class ExampleOutput(BaseModel):
    company: str
    score: int


class FakeBedrockClient:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 12, "outputTokens": 4},
            "stopReason": "end_turn",
        }


def test_bedrock_structured_output_uses_converse_and_validates_json(monkeypatch):
    client = FakeBedrockClient('{"company":"BH Global","score":91}')
    monkeypatch.setattr(aws_llm.BedrockChat, "_client", lambda self: client)

    model = aws_llm.BedrockChat(
        model_id="amazon.nova-lite-v1:0",
        max_tokens=800,
    ).with_structured_output(ExampleOutput)
    result = model.invoke("Assess this role")

    assert result == ExampleOutput(company="BH Global", score=91)
    call = client.calls[0]
    assert call["modelId"] == "amazon.nova-lite-v1:0"
    assert call["messages"][0]["content"][0]["text"] == "Assess this role"
    assert call["inferenceConfig"]["maxTokens"] == 800
    assert "JSON schema" in call["system"][0]["text"]


def test_plain_bedrock_response_preserves_usage(monkeypatch):
    client = FakeBedrockClient("SimplyNext Bedrock ready")
    monkeypatch.setattr(aws_llm.BedrockChat, "_client", lambda self: client)

    result = aws_llm.BedrockChat(model_id="test-model").invoke("ping")

    assert result.content == "SimplyNext Bedrock ready"
    assert result.response_metadata["usage"]["inputTokens"] == 12
    assert result.response_metadata["model_id"] == "test-model"


def test_task_specific_model_overrides_default(monkeypatch):
    monkeypatch.setenv("AWS_BEDROCK_MODEL_ID", "default-model")
    monkeypatch.setenv("AWS_BEDROCK_STAGE2_MODEL_ID", "ranking-model")

    model = aws_llm.build_bedrock_chat(task_model_env="AWS_BEDROCK_STAGE2_MODEL_ID")

    assert model.model_id == "ranking-model"


def test_json_parser_accepts_fenced_or_prefixed_json():
    assert aws_llm._json_from_text('```json\n{"score": 88}\n```') == {"score": 88}
    assert aws_llm._json_from_text('Result: {"score": 89}') == {"score": 89}
