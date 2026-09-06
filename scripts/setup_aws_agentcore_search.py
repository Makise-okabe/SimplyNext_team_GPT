from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


REGION = "us-east-1"
ROLE_NAME = "SimplyNextAgentCoreWebSearchRole"
ROLE_POLICY_NAME = "SimplyNextAgentCoreWebSearchPolicy"
GATEWAY_NAME = "simplenext-career-search"
TARGET_NAME = "simplenext-web-search"
MCP_PROTOCOL_VERSION = "2026-07-28"


def _items(payload: dict, *keys: str) -> list[dict]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _wait(client, method: str, identifier_name: str, identifier: str) -> dict:
    getter = getattr(client, method)
    for _ in range(30):
        payload = getter(**{identifier_name: identifier})
        status = str(payload.get("status") or "").upper()
        if status in {"READY", "ACTIVE"}:
            return payload
        if status in {"FAILED", "UPDATE_UNSUCCESSFUL", "DELETING", "DELETE_FAILED"}:
            raise RuntimeError(f"AWS resource entered {status}: {payload}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {identifier}")


def _wait_target(client, gateway_id: str, target_id: str) -> dict:
    for _ in range(30):
        payload = client.get_gateway_target(
            gatewayIdentifier=gateway_id,
            targetId=target_id,
        )
        status = str(payload.get("status") or "").upper()
        if status in {"READY", "ACTIVE"}:
            return payload
        if status in {
            "FAILED", "UPDATE_UNSUCCESSFUL", "SYNCHRONIZE_UNSUCCESSFUL",
            "DELETING", "DELETE_FAILED",
        }:
            raise RuntimeError(f"AWS target entered {status}: {payload}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for target {target_id}")


def _find_gateway(client) -> dict | None:
    token = None
    while True:
        kwargs = {"maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        payload = client.list_gateways(**kwargs)
        for item in _items(payload, "items", "gateways", "gatewaySummaries"):
            if item.get("name") == GATEWAY_NAME:
                return item
        token = payload.get("nextToken")
        if not token:
            return None


def _find_target(client, gateway_id: str) -> dict | None:
    token = None
    while True:
        kwargs = {"gatewayIdentifier": gateway_id, "maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        payload = client.list_gateway_targets(**kwargs)
        for item in _items(payload, "items", "targets", "gatewayTargetSummaries"):
            if item.get("name") == TARGET_NAME:
                return item
        token = payload.get("nextToken")
        if not token:
            return None


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise SystemExit("AWS support is not installed. Run: uv sync --extra dev") from exc

    profile = os.getenv("AWS_PROFILE", "").strip() or None
    session = boto3.Session(profile_name=profile, region_name=REGION)
    sts = session.client("sts", region_name=REGION)
    iam = session.client("iam", region_name=REGION)
    agentcore = session.client("bedrock-agentcore-control", region_name=REGION)

    try:
        account_id = sts.get_caller_identity()["Account"]
    except Exception as exc:
        hint = f" --profile {profile}" if profile else ""
        raise SystemExit(f"AWS login failed. Run: aws sso login{hint}\n{exc}") from exc

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {
                    "aws:SourceArn": f"arn:aws:bedrock-agentcore:{REGION}:{account_id}:gateway/*"
                },
            },
        }],
    }
    permission_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeGateway",
                "Resource": f"arn:aws:bedrock-agentcore:{REGION}:{account_id}:gateway/*",
            },
            {
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeWebSearch",
                "Resource": f"arn:aws:bedrock-agentcore:{REGION}:aws:tool/web-search.v1",
            },
        ],
    }

    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        iam.update_assume_role_policy(
            RoleName=ROLE_NAME,
            PolicyDocument=json.dumps(trust_policy),
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=ROLE_NAME,
            Description="AgentCore service role for SimplyNext managed web search",
            AssumeRolePolicyDocument=json.dumps(trust_policy),
        )["Role"]
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=ROLE_POLICY_NAME,
        PolicyDocument=json.dumps(permission_policy),
    )

    gateway = _find_gateway(agentcore)
    if gateway is None:
        time.sleep(5)
        gateway = agentcore.create_gateway(
            name=GATEWAY_NAME,
            roleArn=role["Arn"],
            protocolType="MCP",
            protocolConfiguration={
                "mcp": {"supportedVersions": [MCP_PROTOCOL_VERSION]}
            },
            authorizerType="AWS_IAM",
        )
    gateway_id = gateway.get("gatewayId") or gateway.get("gatewayIdentifier")
    if not gateway_id:
        raise RuntimeError(f"AWS returned no gateway ID: {gateway}")
    gateway = _wait(agentcore, "get_gateway", "gatewayIdentifier", gateway_id)
    supported_versions = (
        gateway.get("protocolConfiguration", {}).get("mcp", {}).get("supportedVersions", [])
    )
    if MCP_PROTOCOL_VERSION not in supported_versions:
        agentcore.update_gateway(
            gatewayIdentifier=gateway_id,
            name=gateway["name"],
            roleArn=gateway["roleArn"],
            protocolType="MCP",
            protocolConfiguration={
                "mcp": {"supportedVersions": [MCP_PROTOCOL_VERSION]}
            },
            authorizerType=gateway["authorizerType"],
        )
        gateway = _wait(agentcore, "get_gateway", "gatewayIdentifier", gateway_id)

    target = _find_target(agentcore, gateway_id)
    if target is None:
        target = agentcore.create_gateway_target(
            name=TARGET_NAME,
            gatewayIdentifier=gateway_id,
            targetConfiguration={
                "mcp": {
                    "connector": {
                        "source": {"connectorId": "web-search", "version": "1.2.0"},
                        "configurations": [{
                            "name": "WebSearch",
                            "parameterValues": {
                                "domainFilter": {
                                    "exclude": ["google.com", "bing.com", "duckduckgo.com"]
                                }
                            },
                        }],
                    }
                }
            },
            credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}],
        )
    target_id = target.get("targetId") or target.get("gatewayTargetId") or target.get("targetIdentifier")
    if target_id:
        _wait_target(agentcore, gateway_id, target_id)

    gateway_url = gateway.get("gatewayUrl") or gateway.get("endpoint")
    if not gateway_url:
        gateway_url = agentcore.get_gateway(gatewayIdentifier=gateway_id).get("gatewayUrl")
    if not gateway_url:
        raise RuntimeError("Gateway is ready but AWS returned no gatewayUrl")

    print("AWS AgentCore Web Search is configured.")
    print("Add these non-secret values to .env:")
    print(f"AWS_AGENTCORE_REGION={REGION}")
    print(f"AWS_AGENTCORE_MCP_VERSION={MCP_PROTOCOL_VERSION}")
    print(f"AWS_AGENTCORE_GATEWAY_URL={gateway_url}")
    print(f"AWS_AGENTCORE_WEB_SEARCH_TOOL={TARGET_NAME}___WebSearch")
    print("Then run: uv run python scripts/check_aws_access.py")


if __name__ == "__main__":
    main()
