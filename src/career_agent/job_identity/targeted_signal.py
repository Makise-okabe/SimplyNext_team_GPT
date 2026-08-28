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
GENERIC_COMPANY_WORDS = {
    "pte",
    "ltd",
    "limited",
    "inc",
    "asia",
    "singapore",
    "company",
    "corporation",
    "corp",
}


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


def _company_tokens(company: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (company or "").lower())
        if len(token) >= 3 and token not in GENERIC_COMPANY_WORDS
    }


def _is_company_url(url: str, company: str | None) -> bool:
    try:
        host_tokens = set(re.findall(r"[a-z0-9]+", urlparse(url).netloc.lower()))
    except ValueError:
        return False
    return bool(_company_tokens(company) & host_tokens)


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


def _nearest_company_link(html: str, title: str, company: str) -> list[str]:
    """Return only the company link nearest to the exact title in DOM order."""
    if not html or not title or not company:
        return []

    soup = BeautifulSoup(html, "html.parser")
    title_lower = title.lower()
    title_node = soup.find(string=lambda value: bool(value and title_lower in value.lower()))
    if title_node is None:
        return []

    descendants = list(soup.descendants)
    positions = {id(node): index for index, node in enumerate(descendants)}
    title_position = positions.get(id(title_node))
    if title_position is None:
        return []

    ranked: list[tuple[int, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or not _is_company_url(href, company):
            continue
        position = positions.get(id(anchor))
        if position is None:
            continue
        ranked.append((abs(position - title_position), href))

    if not ranked:
        return []
    ranked.sort(key=lambda item: item[0])
    return [ranked[0][1]]


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
    return _dedupe([url for url in email.links if _is_company_url(url, company)])


def _infer_type(context: str) -> str:
    lowered = context.lower()
    if any(marker in lowered for marker in INTERNSHIP_MARKERS):
        return "internship"
    if any(marker in lowered for marker in FULL_TIME_MARKERS):
        return "full_time"
    return "unknown"


def _infer_type_from_section(email: EmailMessage, title: str) -> str:
    """Infer JOBS vs INTERNSHIPS from the nearest preceding newsletter section.

    Dense CFG newsletters often omit `Full-Time Job` from an individual row. The
    surrounding section is still explicit source evidence and is more reliable
    than guessing from the role title.
    """
    text = email.body_text or ""
    index = text.lower().find(title.lower())
    if index < 0:
        return "unknown"

    prefix = text[:index].lower()
    jobs_positions = [
        prefix.rfind("\njobs"),
        prefix.rfind(" jobs "),
        prefix.rfind("**jobs**"),
    ]
    internship_positions = [
        prefix.rfind("\ninternships"),
        prefix.rfind(" internships "),
        prefix.rfind("**internships"),
    ]
    last_jobs = max(jobs_positions)
    last_internships = max(internship_positions)

    if last_jobs < 0 and last_internships < 0:
        return "unknown"
    return "full_time" if last_jobs > last_internships else "internship"


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

    urls = row_links
    if not urls:
        urls = _nearest_company_link(html, title, company)
    if not urls:
        urls = _company_host_links(email, company)[:1]

    opportunity_type = _infer_type(context)
    if opportunity_type == "unknown":
        opportunity_type = _infer_type_from_section(email, title)

    return OpportunitySignal(
        source_type="outlook",
        source_name=email.sender_name or email.sender_email or "NUS career email",
        source_message_id=email.message_id,
        source_date=email.received_at,
        company=company,
        role_title=title,
        location=_infer_location(context),
        opportunity_type=opportunity_type,
        urls=urls,
        raw_text=context[:2200],
        resolution_status="unresolved",
    )
