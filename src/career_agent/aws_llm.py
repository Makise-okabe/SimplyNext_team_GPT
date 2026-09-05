from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel


DEFAULT_BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"
DEFAULT_BEDROCK_REGION = "us-east-1"


@dataclass(frozen=True)
class BedrockMessage:
    content: str
    response_metadata: dict[str, Any]


def aws_region() -> str:
    load_dotenv()
    return (
        os.getenv("AWS_BEDROCK_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or DEFAULT_BEDROCK_REGION
    ).strip()


def bedrock_model_id(task_env: str | None = None) -> str:
    load_dotenv()
    if task_env:
        task_model = os.getenv(task_env, "").strip()
        if task_model:
            return task_model
    return os.getenv("AWS_BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID).strip() or DEFAULT_BEDROCK_MODEL_ID


def _json_from_text(text: str) -> Any:
    value = (text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Amazon Bedrock returned no JSON object")
        return json.loads(value[start : end + 1])


def _response_text(response: dict[str, Any]) -> str:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    text = "\n".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
    if not text.strip():
        raise RuntimeError("Amazon Bedrock returned an empty response")
    return text.strip()


@dataclass(frozen=True)
class BedrockChat:
    model_id: str
    temperature: float = 0.0
    timeout: float = 30.0
    max_retries: int = 1
    max_tokens: int = 4096
    output_schema: type[BaseModel] | None = None

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> "BedrockChat":
        return replace(self, output_schema=schema)

    def _client(self):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("AWS support is not installed; run `uv sync --extra dev`") from exc

        load_dotenv()
        profile = os.getenv("AWS_PROFILE", "").strip() or None
        try:
            session = boto3.Session(profile_name=profile, region_name=aws_region())
            return session.client(
                "bedrock-runtime",
                config=Config(
                    connect_timeout=min(self.timeout, 10.0),
                    read_timeout=self.timeout,
                    retries={"max_attempts": max(self.max_retries + 1, 1), "mode": "standard"},
                ),
            )
        except Exception as exc:
            profile_hint = f" --profile {profile}" if profile else ""
            raise RuntimeError(
                "AWS credentials are unavailable. Run `aws sso login"
                f"{profile_hint}` and retry."
            ) from exc

    def invoke(self, prompt: str) -> BaseModel | BedrockMessage:
        system_text = (
            "Follow the user's extraction and assessment instructions exactly. "
            "Do not invent facts, requirements, companies, roles, URLs, or evidence."
        )
        if self.output_schema is not None:
            schema = json.dumps(self.output_schema.model_json_schema(), ensure_ascii=False)
            system_text += (
                " Return only one valid JSON object that conforms to this JSON schema. "
                "Do not use Markdown fences or add commentary. JSON schema: " + schema
            )

        try:
            response = self._client().converse(
                modelId=self.model_id,
                system=[{"text": system_text}],
                messages=[{"role": "user", "content": [{"text": str(prompt)}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": 0.9,
                },
            )
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            if "ExpiredToken" in message or "UnauthorizedSSOToken" in message:
                raise RuntimeError("AWS SSO session expired; run `aws sso login` again") from exc
            raise RuntimeError(f"Amazon Bedrock invocation failed ({name}): {message}") from exc

        text = _response_text(response)
        if self.output_schema is not None:
            return self.output_schema.model_validate(_json_from_text(text))
        return BedrockMessage(
            content=text,
            response_metadata={
                "usage": response.get("usage", {}),
                "stopReason": response.get("stopReason"),
                "model_id": self.model_id,
            },
        )


def build_bedrock_chat(
    *,
    task_model_env: str | None = None,
    temperature: float = 0.0,
    timeout: float = 30.0,
    max_retries: int = 1,
    max_tokens: int = 4096,
) -> BedrockChat:
    return BedrockChat(
        model_id=bedrock_model_id(task_model_env),
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )
