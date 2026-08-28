from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from career_agent.models.job_v4 import LivenessResult

CLOSED_MARKERS = (
    "job is no longer available",
    "job no longer available",
    "position is no longer available",
    "position has been filled",
    "vacancy has been filled",
    "no longer accepting applications",
    "this job has expired",
    "job has expired",
    "posting has expired",
)
OPEN_MARKERS = (
    "apply now",
    "apply for this job",
    "apply for this position",
    "submit application",
    "start application",
)


def classify_liveness(status_code: int, text: str) -> tuple[str, str]:
    normalized = " ".join((text or "").lower().split())

    if status_code in {404, 410}:
        return "closed", f"HTTP {status_code}"
    if any(marker in normalized for marker in CLOSED_MARKERS):
        return "closed", "explicit closed/expired marker found"
    if 200 <= status_code < 300 and any(marker in normalized for marker in OPEN_MARKERS):
        return "open", "active application marker found"
    if 200 <= status_code < 300:
        return "unknown", "page reachable but no reliable open/closed marker"
    return "unknown", f"HTTP {status_code} does not prove liveness"


def _looks_like_generic_redirect(requested_url: str, final_url: str) -> bool:
    requested = urlparse(requested_url)
    final = urlparse(final_url)
    if requested.netloc.lower() != final.netloc.lower():
        return False
    requested_depth = len([part for part in requested.path.split("/") if part])
    final_parts = [part.lower() for part in final.path.split("/") if part]
    if requested_depth >= 2 and len(final_parts) <= 1:
        return True
    if requested_depth >= 2 and final_parts in (["careers"], ["jobs"]):
        return True
    return False


def check_liveness(url: str, timeout_seconds: float = 6.0) -> LivenessResult:
    """Cheap, conservative refresh of an already-verified official URL."""
    started = time.perf_counter()
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"
    }

    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        final_url = str(response.url)
        content_type = response.headers.get("content-type", "").lower()
        text = ""
        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())[:12000]

        status, reason = classify_liveness(response.status_code, text)
        if status == "open" and _looks_like_generic_redirect(url, final_url):
            status = "unknown"
            reason = "job URL redirected to a generic careers/jobs page"

        return LivenessResult(
            url=url,
            final_url=final_url,
            status=status,
            status_code=response.status_code,
            reason=reason,
            checked_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        return LivenessResult(
            url=url,
            status="unknown",
            reason="liveness check unavailable",
            checked_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            warning=f"{type(exc).__name__}: {exc}",
        )
