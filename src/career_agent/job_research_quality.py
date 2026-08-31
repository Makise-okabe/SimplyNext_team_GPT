from __future__ import annotations

import re
from urllib.parse import urlparse

AGGREGATOR_HOST_MARKERS = (
    "linkedin.com",
    "indeed.",
    "glassdoor.",
    "jobstreet.",
    "jobsdb.",
    "trabajo.org",
    "talent.com",
    "grabjobs.",
    "foundit.",
    "jooble.",
)
ATS_HOST_MARKERS = (
    "myworkdayjobs.com",
    "workdayjobs.com",
    "greenhouse.io",
    "lever.co",
    "smartrecruiters.com",
    "successfactors.com",
    "taleo.net",
    "icims.com",
    "jobvite.com",
    "mokahr.com",
)

CLOSED_MARKERS = (
    "no longer accepting applications",
    "this job is no longer available",
    "this position is no longer available",
    "job has expired",
    "position has been filled",
    "applications are closed",
)

STOP_MARKERS = (
    "\nsimilar jobs\n",
    "\npeople also viewed\n",
    "\nreferrals increase your chances",
    "\nget notified when a new job is posted",
    "\nget notified about new\n",
    "\nexplore top content on linkedin\n",
    "\nlinkedin\n©",
)

DROP_EXACT_LINES = {
    "skip to main content",
    "expand search",
    "jobs",
    "people",
    "learning",
    "clear text",
    "sign in",
    "join now",
    "save",
    "report this job",
    "show more",
    "show less",
    "email or phone",
    "password",
    "forgot password?",
}


def host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().split(":", 1)[0]
    except ValueError:
        return ""


def is_aggregator_url(url: str | None) -> bool:
    value = host(url)
    return bool(value and any(marker in value for marker in AGGREGATOR_HOST_MARKERS))


def is_secondary_url(url: str | None) -> bool:
    return is_aggregator_url(url)


def _company_tokens(company: str | None) -> list[str]:
    stop = {"pte", "ltd", "limited", "inc", "group", "holdings", "singapore", "company"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (company or "").lower())
        if len(token) >= 4 and token not in stop
    ]


def is_plausible_official_url(url: str | None, company: str | None) -> bool:
    """Conservative primary-source gate: employer domain or known ATS, never aggregator."""
    value = host(url)
    if not value or is_aggregator_url(url):
        return False
    if value in {"careers.example.com", "jobs.example.com"}:
        return True
    if any(marker in value for marker in ATS_HOST_MARKERS):
        return True
    compact_host = value.replace("-", "").replace(".", "")
    return any(token.replace("-", "") in compact_host for token in _company_tokens(company))


def page_is_closed(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    return any(marker in lowered for marker in CLOSED_MARKERS)


def clean_jd_text(text: str) -> str:
    """Keep job content while removing common LinkedIn/search-page chrome and recommendations."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lowered = normalized.lower()
    cut = len(normalized)
    for marker in STOP_MARKERS:
        index = lowered.find(marker)
        if index >= 0:
            cut = min(cut, index)
    normalized = normalized[:cut]

    kept: list[str] = []
    previous = None
    for raw in normalized.splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue
        lowered_line = line.lower()
        if lowered_line in DROP_EXACT_LINES:
            continue
        if lowered_line.startswith("by clicking continue to join or sign in"):
            continue
        if lowered_line.startswith("use ai to assess how you fit"):
            continue
        if lowered_line.startswith("get ai-powered advice"):
            continue
        if line == previous:
            continue
        kept.append(line)
        previous = line

    return "\n".join(kept).strip()
