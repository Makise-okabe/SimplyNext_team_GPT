from __future__ import annotations

import os
import argparse
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
    parser = argparse.ArgumentParser(description="Make live Bedrock and AgentCore requests; do not use search cache.")
    parser.add_argument("--query", default='"BH Global" "Electrical Intern"')
    args = parser.parse_args()
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
    print(f"  Usage   : {reply.response_metadata.get('usage', {})}")

    gateway_url = os.getenv("AWS_AGENTCORE_GATEWAY_URL", "").strip()
    if not gateway_url:
        raise SystemExit("Search not configured; run scripts/setup_aws_agentcore_search.py")

    from career_agent.tools.web_search import _search_aws_agentcore

    try:
        results = _search_aws_agentcore(args.query, 8, bypass_cache=True)
    except Exception as exc:
        raise SystemExit(f"AgentCore Web Search test failed: {exc}") from exc
    if not results:
        raise SystemExit("Search returned ZERO usable URLs. Search readiness has NOT been established.")
    print(f"  Search  : live response ({len(results)} result(s)); verify company/title below")
    for result in results:
        print(f"            {result.title} | {result.url}")


if __name__ == "__main__":
    main()
