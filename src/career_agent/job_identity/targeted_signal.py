from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from career_agent.models.email import EmailMessage
from career_agent.models.signal import OpportunitySignal

BASED_IN_PATTERN = re.compile(r"(?i)based\s+in\s+([A-Za-z][A-Za-z .'-]{2,40})")
FULL_TIME_MARKERS = ("full-time", "full time", "graduate", "associate")
INTERNSHIP_MARKERS = ("internship", "intern ", "intern-", " intern")


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _find_target_container(html: str, title: str) -> tuple[str, list[str]]:
    if not html or not title:
        return "", []

    soup = BeautifulSoup(html, "html.parser")
    title_lower = title.lower()
    text_node = soup.find(string=lambda value: bool(value and title_lower in value.lower()))
    if text_node is None:
        return "", []

    node = text_node.parent
    container = node.find_parent("tr") or node.find_parent("li") or node.find_parent("p") or node.parent
    if container is None:
        return "", []

    text = _clean(unescape(container.get_text(" ", strip=True)))
    links = [anchor.get("href", "").strip() for anchor in container.find_all("a", href=True)]
    return text, _dedupe(links)


def _text_window(text: str, title: str, radius: int = 900) -> str:
    if not text or not title:
        return ""
    index = text.lower().find(title.lower())
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(title) + radius)
    return _clean(text[start:end])


def _company_host_links(email: EmailMessage, company: str | None) -> list[str]:
    if not company:
        return []
    company_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", company.lower())
        if len(token) >= 3 and token not in {"pte", "ltd", "limited", "inc", "asia", "singapore"}
    }
    result: list[str] = []
    for url in email.links:
        try:
            host_tokens = set(re.findall(r"[a-z0-9]+", urlparse(url).netloc.lower()))
        except ValueError:
            continue
        if company_tokens & host_tokens:
            result.append(url)
    return _dedupe(result)


def _infer_type(context: str) -> str:
    lowered = context.lower()
    if any(marker in lowered for marker in INTERNSHIP_MARKERS):
        return "internship"
    if any(marker in lowered for marker in FULL_TIME_MARKERS):
        return "full_time"
    return "unknown"


def _infer_location(context: str) -> str | None:
    match = BASED_IN_PATTERN.search(context)
    if not match:
        return None
    value = _clean(match.group(1)).strip(" .,-")
    value = re.split(r"(?i)\b(?:deadline|apply|full-time|full time|ug|pg)\b", value)[0]
    return _clean(value) or None


def build_targeted_signal(
    email: EmailMessage,
    company: str,
    title: str,
) -> OpportunitySignal | None:
    """Deterministically recover one explicitly requested newsletter role.

    Exact title presence is mandatory. The nearest HTML row is preferred so a
    target role inherits only its own hyperlink instead of neighbouring jobs.
    """
    if not company or not title:
        return None

    full_text = email.body_text or ""
    html = email.body_html or ""
    if title.lower() not in full_text.lower() and title.lower() not in html.lower():
        return None

    row_text, row_links = _find_target_container(html, title)
    context = row_text or _text_window(full_text, title)
    if not context:
        return None

    urls = row_links or _company_host_links(email, company)

    return OpportunitySignal(
        source_type="outlook",
        source_name=email.sender_name or email.sender_email or "NUS career email",
        source_message_id=email.message_id,
        source_date=email.received_at,
        company=company,
        role_title=title,
        location=_infer_location(context),
        opportunity_type=_infer_type(context),
        urls=urls,
        raw_text=context[:2200],
        resolution_status="unresolved",
    )
