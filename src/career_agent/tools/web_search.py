from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from career_agent.job_research_quality import is_plausible_official_url

load_dotenv()


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


BING_URL = "https://www.bing.com/search"
BING_RSS_URL = "https://www.bing.com/search?format=rss"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"
SEARCH_TIMEOUT_SECONDS = 6.0
AWS_SEARCH_TIMEOUT_SECONDS = 20.0
AWS_AGENTCORE_MCP_VERSION = "2026-07-28"
DEFAULT_AGENTCORE_MAX_CALLS = 15
DEFAULT_AGENTCORE_CACHE_TTL_SECONDS = 12 * 60 * 60
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WEB_SEARCH_MODEL = "groq/compound-mini"
DEFAULT_GROQ_SEARCH_MAX_CALLS = 15
LOGGER = logging.getLogger(__name__)
_AGENTCORE_BUDGET_LOCK = Lock()
_AGENTCORE_NETWORK_CALLS = 0
_AGENTCORE_BUDGET_WARNING_EMITTED = False
_GROQ_BUDGET_LOCK = Lock()
_GROQ_NETWORK_CALLS = 0
_GROQ_BUDGET_WARNING_EMITTED = False
SITE_PATTERN = re.compile(r"(?i)(?:^|\s)site:([^\s\"']+)")
QUOTED_PATTERN = re.compile(r'"([^\"]+)"')
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
QUERY_STOPWORDS = {
    "the", "and", "for", "with", "job", "jobs", "career", "careers",
    "role", "position", "singapore", "pte", "ltd", "limited", "private",
}


def stable_search_api_name() -> str | None:
    """Return the configured stable web-search API, if any."""
    if os.getenv("AWS_AGENTCORE_GATEWAY_URL", "").strip():
        return "aws_agentcore_web_search"
    return None


def _unwrap_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    return unquote(target) if target else href


def _decode_bing_target(value: str) -> str | None:
    target = unquote(value or "").strip()
    if target.startswith(("http://", "https://")):
        return target
    if target.startswith("a1") and len(target) > 4:
        payload = target[2:]
        payload += "=" * ((4 - len(payload) % 4) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
        if decoded.startswith(("http://", "https://")):
            return decoded
    return None


def _unwrap_bing_url(href: str) -> str:
    parsed = urlparse(href)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"bing.com", "www.bing.com"}:
        return href
    query = parse_qs(parsed.query)
    for key in ("u", "url", "r"):
        for value in query.get(key, []):
            decoded = _decode_bing_target(value)
            if decoded:
                return decoded
    return href


def _normalize_result_url(href: str) -> str:
    value = _unwrap_duckduckgo_url(href.strip())
    return _unwrap_bing_url(value)


def _normalize_host(value: str) -> str:
    host_value = (value or "").lower().split(":", 1)[0]
    return host_value[4:] if host_value.startswith("www.") else host_value


def _site_constraint(query: str) -> tuple[str, str] | None:
    match = SITE_PATTERN.search(query or "")
    if not match:
        return None
    raw = match.group(1).strip().rstrip("/")
    parsed = urlparse("https://" + raw)
    target_host = _normalize_host(parsed.hostname or "")
    target_path = parsed.path.rstrip("/")
    if not target_host:
        return None
    return target_host, target_path


def _relax_site_query(query: str) -> str:
    match = SITE_PATTERN.search(query or "")
    if not match:
        return query
    target = match.group(1).strip().rstrip("/")
    return re.sub(SITE_PATTERN, f" {target}", query, count=1).strip()


def _agentcore_arguments(query: str, max_results: int) -> dict:
    """Translate search syntax into the AgentCore connector's native schema.

    AgentCore does not document Google-style ``site:`` operators. Connector
    v1.2.0 exposes a real domain include filter, so send the hostname there and
    keep only the human search terms in ``query``.
    """
    arguments: dict = {"query": query[:200], "maxResults": max(1, min(max_results, 25))}
    constraint = _site_constraint(query)
    if constraint is None:
        return arguments
    target_host, _ = constraint
    search_terms = SITE_PATTERN.sub(" ", query, count=1)
    search_terms = re.sub(r"\s+", " ", search_terms).strip() or target_host
    arguments["query"] = search_terms[:200]
    arguments["filters"] = {"domainFilter": {"include": [target_host]}}
    return arguments


def _result_matches_site(result: SearchResult, constraint: tuple[str, str] | None) -> bool:
    if constraint is None:
        return True
    target_host, target_path = constraint
    parsed = urlparse(result.url)
    result_host = _normalize_host(parsed.hostname or "")
    if result_host != target_host and not result_host.endswith("." + target_host):
        return False
    if target_path:
        path = parsed.path.rstrip("/")
        return path == target_path or path.startswith(target_path + "/")
    return True


def _apply_site_constraint(
    results: list[SearchResult],
    constraint: tuple[str, str] | None,
    max_results: int,
) -> list[SearchResult]:
    if constraint is None:
        return results[:max_results]
    return [result for result in results if _result_matches_site(result, constraint)][:max_results]


def _query_tokens(value: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall((value or "").lower())
        if len(token) >= 3 and token not in QUERY_STOPWORDS
    }


def _quoted_query_parts(query: str) -> list[str]:
    return [part.strip() for part in QUOTED_PATTERN.findall(query or "") if part.strip()]


def _result_matches_quoted_query(result: SearchResult, query: str) -> bool:
    parts = _quoted_query_parts(query)
    if len(parts) < 2:
        return True

    text = f"{result.title} {result.snippet} {result.url}".lower()
    text_tokens = _query_tokens(text)

    company_tokens = _query_tokens(parts[0])
    title_tokens = _query_tokens(parts[1])

    company_match = (
        not company_tokens
        or bool(company_tokens & text_tokens)
        or is_plausible_official_url(result.url, parts[0])
    )
    if not company_match:
        compact_company = "".join(sorted(company_tokens))
        compact_text = re.sub(r"[^a-z0-9]", "", text)
        company_match = bool(compact_company and compact_company in compact_text)

    if not title_tokens:
        title_match = parts[1].lower() in text
    else:
        overlap = len(title_tokens & text_tokens) / len(title_tokens)
        title_match = overlap >= 0.50

    return company_match and title_match


def _apply_query_relevance(
    results: list[SearchResult],
    query: str,
    max_results: int,
) -> list[SearchResult]:
    return [result for result in results if _result_matches_quoted_query(result, query)][:max_results]


def _simplify_query(query: str) -> str:
    value = query.replace('"', " ").replace("'", " ")
    value = re.sub(r"[(),|]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _append_result(
    results: list[SearchResult],
    seen: set[str],
    *,
    title: str,
    href: str,
    snippet: str = "",
) -> None:
    url = _normalize_result_url(href)
    if not url.startswith(("http://", "https://")) or url in seen:
        return
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"bing.com", "www.bing.com"} and parsed.path.startswith("/ck/"):
        return
    seen.add(url)
    results.append(
        SearchResult(
            title=" ".join(title.split()),
            url=url,
            snippet=" ".join(snippet.split()),
        )
    )


def _agentcore_gateway_url() -> str:
    value = os.getenv("AWS_AGENTCORE_GATEWAY_URL", "").strip().rstrip("/")
    if value and not value.endswith("/mcp"):
        value += "/mcp"
    return value


def _agentcore_json_response(response: httpx.Response) -> dict:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        return response.json()
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("AgentCore Gateway returned no JSON-RPC payload")


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _claim_groq_network_call() -> bool:
    global _GROQ_NETWORK_CALLS, _GROQ_BUDGET_WARNING_EMITTED
    limit = _positive_int_env("SIMPLYNEXT_GROQ_SEARCH_MAX_CALLS", DEFAULT_GROQ_SEARCH_MAX_CALLS)
    with _GROQ_BUDGET_LOCK:
        if _GROQ_NETWORK_CALLS >= limit:
            if not _GROQ_BUDGET_WARNING_EMITTED:
                LOGGER.warning("Groq grounded-search budget reached (%s call(s))", limit)
                _GROQ_BUDGET_WARNING_EMITTED = True
            return False
        _GROQ_NETWORK_CALLS += 1
        return True


def _groq_search_result_items(message: dict) -> list[dict]:
    items: list[dict] = []
    for tool in message.get("executed_tools") or []:
        if not isinstance(tool, dict):
            continue
        search_results = tool.get("search_results") or {}
        if isinstance(search_results, dict):
            items.extend(item for item in search_results.get("results") or [] if isinstance(item, dict))
    return items


def search_groq_grounded(query: str, max_results: int = 8) -> list[SearchResult]:
    """One token-capped server-side search used only after normal providers fail."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or not query.strip() or not _claim_groq_network_call():
        return []
    response = httpx.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("GROQ_WEB_SEARCH_MODEL", GROQ_WEB_SEARCH_MODEL).strip() or GROQ_WEB_SEARCH_MODEL,
            "messages": [{
                "role": "user",
                "content": (
                    "Search the live web for this exact job. Return sources for the exact employer and role. "
                    "Prioritize the employer job page, then MyCareersFuture, LinkedIn, JobStreet, Foundit or Glints. "
                    "Do not substitute a similarly named company or a different role. Query: " + query
                ),
            }],
            "compound_custom": {"tools": {"enabled_tools": ["web_search"]}},
            "search_settings": {"country": "singapore"},
            "temperature": 0,
        },
        timeout=20.0,
    )
    if response.is_error:
        detail = " ".join(response.text.split())[:1000] or "<empty response>"
        raise RuntimeError(f"Groq grounded search HTTP {response.status_code}: {detail}")
    payload = response.json()
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in _groq_search_result_items(message or {}):
        _append_result(
            results,
            seen,
            title=str(item.get("title") or ""),
            href=str(item.get("url") or ""),
            snippet=str(item.get("content") or item.get("text") or ""),
        )
        if len(results) >= max_results:
            break
    LOGGER.warning("Groq grounded web search: %s URL result(s) for %s", len(results), query)
    return results


def _agentcore_cache_file(
    gateway_url: str,
    tool_name: str,
    query: str,
    max_results: int,
) -> Path:
    root = Path(os.getenv("SIMPLYNEXT_AGENTCORE_CACHE_DIR", ".cache/agentcore_search_v1"))
    key = json.dumps(
        [gateway_url, tool_name, query, max_results],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def _cached_agentcore_results(path: Path) -> list[SearchResult] | None:
    ttl = _positive_int_env(
        "SIMPLYNEXT_AGENTCORE_CACHE_TTL_SECONDS",
        DEFAULT_AGENTCORE_CACHE_TTL_SECONDS,
    )
    if ttl <= 0 or not path.exists() or time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [SearchResult(**item) for item in payload.get("results", [])]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_agentcore_results(path: Path, results: list[SearchResult]) -> None:
    if _positive_int_env("SIMPLYNEXT_AGENTCORE_CACHE_TTL_SECONDS", DEFAULT_AGENTCORE_CACHE_TTL_SECONDS) <= 0:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"results": [item.__dict__ for item in results]}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        LOGGER.debug("Could not write AgentCore search cache", exc_info=True)


def _claim_agentcore_network_call() -> bool:
    global _AGENTCORE_NETWORK_CALLS, _AGENTCORE_BUDGET_WARNING_EMITTED
    limit = _positive_int_env("SIMPLYNEXT_AGENTCORE_MAX_CALLS", DEFAULT_AGENTCORE_MAX_CALLS)
    with _AGENTCORE_BUDGET_LOCK:
        if _AGENTCORE_NETWORK_CALLS >= limit:
            if not _AGENTCORE_BUDGET_WARNING_EMITTED:
                LOGGER.warning(
                    "AgentCore search budget reached (%s network attempt(s)); using public providers/cache only",
                    limit,
                )
                _AGENTCORE_BUDGET_WARNING_EMITTED = True
            return False
        _AGENTCORE_NETWORK_CALLS += 1
        return True


def _search_aws_agentcore(query: str, max_results: int, *, bypass_cache: bool = False) -> list[SearchResult]:
    """Call the AWS AgentCore managed Web Search connector through an IAM gateway."""
    gateway_url = _agentcore_gateway_url()
    if not gateway_url:
        return []

    try:
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
    except ImportError as exc:
        raise RuntimeError("AWS support is not installed; run `uv sync --extra dev`") from exc

    region = os.getenv("AWS_AGENTCORE_REGION", "us-east-1").strip() or "us-east-1"
    profile = os.getenv("AWS_PROFILE", "").strip() or None
    tool_name = (
        os.getenv("AWS_AGENTCORE_WEB_SEARCH_TOOL", "simplenext-web-search___WebSearch").strip()
        or "simplenext-web-search___WebSearch"
    )
    cache_path = _agentcore_cache_file(gateway_url, tool_name, query, max_results)
    cached = None if bypass_cache else _cached_agentcore_results(cache_path)
    if cached is not None:
        LOGGER.info("AgentCore search cache hit: %s", query)
        return cached[:max_results]
    if not _claim_agentcore_network_call():
        return []
    protocol_version = (
        os.getenv("AWS_AGENTCORE_MCP_VERSION", AWS_AGENTCORE_MCP_VERSION).strip()
        or AWS_AGENTCORE_MCP_VERSION
    )
    arguments = _agentcore_arguments(query, max_results)
    request_meta = {
        "io.modelcontextprotocol/protocolVersion": protocol_version,
        "io.modelcontextprotocol/clientInfo": {
            "name": "simplenext-career-agent",
            "version": "0.1.0",
        },
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    rpc_payload = {
        "jsonrpc": "2.0",
        "id": "simplenext-web-search",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
            "_meta": request_meta,
        },
    }
    body = json.dumps(rpc_payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": protocol_version,
        "Mcp-Method": "tools/call",
        "Mcp-Name": tool_name,
    }

    session = boto3.Session(profile_name=profile, region_name=region)
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials unavailable; run `aws sso login` and retry")
    request = AWSRequest(method="POST", url=gateway_url, data=body, headers=headers)
    SigV4Auth(credentials.get_frozen_credentials(), "bedrock-agentcore", region).add_auth(request)
    response = httpx.post(
        gateway_url,
        content=body,
        headers=dict(request.headers.items()),
        timeout=AWS_SEARCH_TIMEOUT_SECONDS,
    )
    if response.is_error:
        detail = " ".join(response.text.split())[:1200] or "<empty response>"
        raise RuntimeError(
            f"AgentCore Gateway returned HTTP {response.status_code}: {detail}"
        )
    payload = _agentcore_json_response(response)
    if payload.get("error"):
        raise RuntimeError(f"AgentCore Web Search error: {payload['error']}")

    results = _agentcore_search_results(payload, max_results)
    LOGGER.warning("AgentCore search HTTP %s: %s URL result(s) for %s", response.status_code, len(results), query)
    if results:
        _save_agentcore_results(cache_path, results)
    return results


def _agentcore_search_results(payload: dict, max_results: int) -> list[SearchResult]:
    if payload.get("error"):
        raise RuntimeError(f"AgentCore Web Search error: {payload['error']}")
    result = payload.get("result") or {}
    if result.get("isError"):
        detail = " ".join(str(block.get("text", "")) for block in result.get("content", []) if isinstance(block, dict))
        raise RuntimeError(f"AgentCore tool failed: {detail[:1200]}")
    nested = result.get("structuredContent") or {}
    if not isinstance(nested, dict):
        nested = {}
    for block in result.get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        try:
            candidate = json.loads(block.get("text") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "results" in candidate and "results" not in nested:
            nested = candidate
            break

    if not isinstance(nested, dict) or not isinstance(nested.get("results"), list):
        raise RuntimeError("AgentCore returned an unrecognized search response (missing results list)")
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in nested.get("results") or []:
        if not isinstance(item, dict):
            continue
        _append_result(
            results,
            seen,
            title=str(item.get("title") or ""),
            href=str(item.get("url") or ""),
            snippet=str(item.get("text") or item.get("snippet") or ""),
        )
        if len(results) >= max_results:
            break
    return results


def _parse_bing_results(html: str, max_results: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in soup.select("li.b_algo"):
        anchor = item.select_one("h2 a")
        if not anchor:
            continue
        snippet_node = item.select_one(".b_caption p")
        _append_result(
            results,
            seen,
            title=anchor.get_text(" ", strip=True),
            href=anchor.get("href", ""),
            snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
        )
        if len(results) >= max_results:
            break
    return results


def _parse_bing_rss(xml_text: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[str] = set()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        description = item.findtext("description") or ""
        _append_result(results, seen, title=title, href=link, snippet=description)
        if len(results) >= max_results:
            break
    return results


def _parse_html_results(html: str, max_results: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()
    for result in soup.select(".result"):
        anchor = result.select_one(".result__a")
        if not anchor:
            continue
        snippet_node = result.select_one(".result__snippet")
        _append_result(
            results,
            seen,
            title=anchor.get_text(" ", strip=True),
            href=anchor.get("href", ""),
            snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
        )
        if len(results) >= max_results:
            break
    return results


def _parse_lite_results(html: str, max_results: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()
    anchors = soup.select("a.result-link")
    if not anchors:
        anchors = [
            anchor
            for anchor in soup.find_all("a", href=True)
            if anchor.get_text(" ", strip=True)
            and (
                anchor.get("href", "").startswith(("http://", "https://"))
                or "uddg=" in anchor.get("href", "")
            )
        ]
    for anchor in anchors:
        snippet = ""
        parent = anchor.parent
        if parent:
            snippet_node = parent.find_next(class_="result-snippet")
            if snippet_node:
                snippet = snippet_node.get_text(" ", strip=True)
        _append_result(
            results,
            seen,
            title=anchor.get_text(" ", strip=True),
            href=anchor.get("href", ""),
            snippet=snippet,
        )
        if len(results) >= max_results:
            break
    return results


def _request_search(
    url: str,
    query: str,
    *,
    parser,
    max_results: int,
    headers: dict[str, str],
) -> list[SearchResult]:
    response = httpx.get(
        url,
        params={"q": query},
        headers=headers,
        timeout=SEARCH_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    return parser(response.text, max_results)


def _search_variants(query: str, constraint: tuple[str, str] | None) -> list[str]:
    variants: list[str] = [query.strip()]
    simplified = _simplify_query(query)
    if simplified and simplified not in variants:
        variants.append(simplified)
    if constraint is not None:
        relaxed = _relax_site_query(query)
        if relaxed and relaxed not in variants:
            variants.append(relaxed)
        relaxed_simple = _simplify_query(relaxed)
        if relaxed_simple and relaxed_simple not in variants:
            variants.append(relaxed_simple)
    return variants


def _filter_results(
    results: list[SearchResult],
    *,
    original_query: str,
    constraint: tuple[str, str] | None,
    max_results: int,
) -> list[SearchResult]:
    constrained = _apply_site_constraint(results, constraint, max_results)
    if constraint is not None:
        return constrained
    return _apply_query_relevance(constrained, original_query, max_results)


def search_public_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search AWS plus an independent public provider for candidate recall.

    AgentCore is intentionally capped below ``max_results``. A semantically
    plausible but wrong brand expansion (for example BH -> Baker Hughes) must
    not prevent an exact employer page found by Bing/RSS from reaching the
    downstream company/title/page verifier.
    """
    if not query.strip():
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-SG,en;q=0.9",
    }
    constraint = _site_constraint(query)
    variants = _search_variants(query, constraint)
    collected: list[SearchResult] = []
    seen: set[str] = set()

    def merge(items: list[SearchResult]) -> None:
        for item in items:
            if item.url in seen:
                continue
            seen.add(item.url)
            collected.append(item)

    if _agentcore_gateway_url():
        for variant in variants[:1]:
            try:
                aws_results = _filter_results(
                    _search_aws_agentcore(variant, max(1, min(2, max_results // 3))),
                    original_query=query,
                    constraint=constraint,
                    max_results=max(1, min(2, max_results // 3)),
                )
                if aws_results:
                    merge(aws_results)
                    break
            except Exception as exc:
                LOGGER.warning("AWS AgentCore web search failed (%s): %s", type(exc).__name__, exc)
                break

    # A site-scoped result already passed a hard domain filter, so another
    # provider adds little value. Unscoped job discovery always gets an
    # independent provider chance before returning AWS candidates.
    if constraint is not None and collected:
        return collected[:max_results]

    providers = (
        (BING_URL, _parse_bing_results),
        (BING_RSS_URL, _parse_bing_rss),
        (DUCKDUCKGO_HTML_URL, _parse_html_results),
        (DUCKDUCKGO_LITE_URL, _parse_lite_results),
    )
    for variant in variants:
        for url, parser in providers:
            try:
                raw_results = _request_search(
                    url,
                    variant,
                    parser=parser,
                    max_results=max_results,
                    headers=headers,
                )
                results = _filter_results(
                    raw_results,
                    original_query=query,
                    constraint=constraint,
                    max_results=max_results,
                )
                if results:
                    merge(results)
                    return collected[:max_results]
            except Exception:
                continue
    return collected[:max_results]
    return []
