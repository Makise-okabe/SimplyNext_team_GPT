from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.aws_llm import aws_region, bedrock_model_id, build_bedrock_chat


def main() -> None:
    load_dotenv()
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("AWS support is not installed. Run: uv sync --extra dev") from exc

    profile = os.getenv("AWS_PROFILE", "").strip() or None
    region = aws_region()
    session = boto3.Session(profile_name=profile, region_name=region)

    try:
        identity = session.client("sts").get_caller_identity()
    except Exception as exc:
        hint = f" --profile {profile}" if profile else ""
        raise SystemExit(f"AWS login failed. Run: aws sso login{hint}\n{type(exc).__name__}: {exc}") from exc

    print("AWS identity ready")
    print(f"  account : {identity.get('Account')}")
    print(f"  region  : {region}")
    print(f"  profile : {profile or '<default credential chain>'}")
    print(f"  model   : {bedrock_model_id()}")

    try:
        reply = build_bedrock_chat(max_tokens=32, timeout=30, max_retries=0).invoke(
            "Reply with exactly: SimplyNext Bedrock ready"
        )
    except Exception as exc:
        raise SystemExit(f"Bedrock test failed: {exc}") from exc
    print(f"  Bedrock : {reply.content}")

    gateway_url = os.getenv("AWS_AGENTCORE_GATEWAY_URL", "").strip()
    if not gateway_url:
        print("  Search  : not configured; run scripts/setup_aws_agentcore_search.py")
        return

    from career_agent.tools.web_search import _search_aws_agentcore

    try:
        results = _search_aws_agentcore("Amazon software engineer careers", 2)
    except Exception as exc:
        raise SystemExit(f"AgentCore Web Search test failed: {exc}") from exc
    print(f"  Search  : ready ({len(results)} result(s))")
    for result in results:
        print(f"            {result.title} | {result.url}")


if __name__ == "__main__":
    main()
