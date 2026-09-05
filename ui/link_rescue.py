from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from career_agent.job_research_quality import is_plausible_official_url
from career_agent.tools.web_search import SearchResult, search_public_web

DATE_NOISE_PAREN_PATTERN = re.compile(
    r"\([^)]*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|20\d{2})[^)]*\)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TITLE_STOPWORDS = {
    "the", "and", "for", "with", "role", "position", "hiring", "career",
    "careers", "job", "jobs", "singapore",
}
COMPANY_STOPWORDS = {
    "the", "pte", "ltd", "limited", "inc", "private", "company", "corporation",
    "corp", "plc", "llc", "singapore", "branch", "holdings", "group",
}
TOKEN_CANONICAL = {
    "engineers": "engineer",
    "developer": "develop",
    "developers": "develop",
    "development": "develop",
    "developing": "develop",
    "internship": "intern",
    "internships": "intern",
    "researcher": "research",
    "researchers": "research",
    "applications": "application",
    "designing": "design",
    "designer": "design",
    "designers": "design",
    "analytics": "analysis",
    "analyst": "analysis",
    "analysts": "analysis",
}
ATS_HOST_MARKERS = (
    "myworkdayjobs.com", "workdayjobs.com", "greenhouse.io", "lever.co",
    "smartrecruiters.com", "successfactors.com", "taleo.net", "oraclecloud.com",
    "icims.com", "jobvite.com", "eightfold.ai", "mokahr.com",
)
SECONDARY_HOST_MARKERS = (
    "linkedin.com", "glassdoor.", "glints.com", "jobstreet.", "jobsdb.",
    "mycareersfuture.gov.sg", "indeed.",
)
CONCRETE_JOB_MARKERS = (
    "/job/", "/jobs/", "jobdetail", "job-detail", "job-listing", "jobid=",
    "job_id=", "jobcode=", "requisition", "reqid=", "/position/", "/positions/",
    "/opening/", "/openings/", "/opportunity/", "/opportunities/",
    "linkedin.com/jobs/view",
)


@dataclass(frozen=True)
class RescuedLink:
    url: str
    kind: str
    confidence: str
    query: str
    result_title: str


def clean_title(value: str | None) -> str:
    text = DATE_NOISE_PAREN_PATTERN.sub(" ", value or "")
    return " ".join(text.split()).strip()


def _canonical(token: str) -> str:
    return TOKEN_CANONICAL.get(token.lower(), token.lower())


def _title_tokens(value: str | None) -> set[str]:
    values = set()
    for raw in TOKEN_PATTERN.findall(clean_title(value).lower()):
        if raw in TITLE_STOPWORDS:
            continue
        if len(raw) < 2:
            continue
        values.add(_canonical(raw))
    return values


def _title_overlap(title: str | None, text: str) -> float:
    source = _title_tokens(title)
    if not source:
        return 0.0
    target = _title_tokens(text)
    return len(source & target) / len(source)


def _company_tokens(company: str | None) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall((company or "").lower())
        if token not in COMPANY_STOPWORDS and len(token) >= 2
    ]


def _company_match(company: str | None, text: str) -> bool:
    tokens = _company_tokens(company)
    if not tokens:
        return False
    lowered = (text or "").lower()
    identity_tokens = set(TOKEN_PATTERN.findall(lowered))
    compact = re.sub(r"[^a-z0-9]", "", lowered)

    distinctive = [token for token in tokens if len(token) >= 4]
    if distinctive and any(token in identity_tokens or token in compact for token in distinctive):
        return True

    if len(tokens) >= 2 and all(token in identity_tokens for token in tokens if len(token) <= 3):
        return True

    acronym = "".join(token[0] for token in tokens)
    if len(acronym) >= 2 and acronym in identity_tokens:
        return True

    compact_company = "".join(tokens)
    return len(compact_company) >= 5 and compact_company in compact


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def _is_ats(url: str) -> bool:
    host = _host(url)
    return any(marker in host for marker in ATS_HOST_MARKERS)


def _is_secondary(url: str) -> bool:
    host = _host(url)
    return any(marker in host for marker in SECONDARY_HOST_MARKERS)


def _is_concrete_job_url(url: str) -> bool:
    value = (url or "").lower()
    return any(marker in value for marker in CONCRETE_JOB_MARKERS)


def _score(company: str, title: str, result: SearchResult) -> tuple[float, str, str] | None:
    identity = f"{result.title} {result.snippet} {result.url}"
    if not _company_match(company, identity):
        return None

    overlap = _title_overlap(title, identity)
    if overlap < 0.72:
        return None

    official = is_plausible_official_url(result.url, company) or _is_ats(result.url)
    secondary = _is_secondary(result.url)
    concrete = _is_concrete_job_url(result.url)

    # We only want a role-specific destination. Generic company career pages are
    # intentionally rejected here; they are not a valid answer to "open this job".
    if official and concrete:
        return 200.0 + 100.0 * overlap, "official_exact", "high"
    if secondary and concrete and overlap >= 0.82:
        return 120.0 + 100.0 * overlap, "secondary_exact", "medium"
    return None


def _queries(company: str, title: str) -> list[str]:
    cleaned = clean_title(title)
    values = [
        f'"{company}" "{cleaned}" careers job',
        f'"{company}" "{cleaned}" apply',
        f'"{company}" "{cleaned}" Singapore',
        f'site:linkedin.com/jobs/view "{company}" "{cleaned}"',
    ]
    seen: set[str] = set()
    return [query for query in values if query and not (query in seen or seen.add(query))]


def rescue_exact_link(company: str, title: str) -> RescuedLink | None:
    """Bounded exact-role recovery: primary first, trusted secondary second."""
    scored: list[tuple[float, SearchResult, str, str, str]] = []
    seen_urls: set[str] = set()

    for query in _queries(company, title):
        try:
            results = search_public_web(query, max_results=10)
        except Exception:
            results = []

        for result in results:
            if not result.url or result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            score = _score(company, title, result)
            if score is None:
                continue
            points, kind, confidence = score
            scored.append((points, result, kind, confidence, query))

        if any(kind == "official_exact" for _, _, kind, _, _ in scored):
            break

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    _, result, kind, confidence, query = scored[0]
    return RescuedLink(
        url=result.url,
        kind=kind,
        confidence=confidence,
        query=query,
        result_title=result.title,
    )


def _has_direct_link(card: dict) -> bool:
    url = str(card.get("job_page_url") or "").strip()
    kind = str(card.get("job_page_kind") or "").lower()
    if url and kind in {"official_exact", "secondary_exact"}:
        return True
    for field in ("official_job_url", "primary_source_url", "secondary_source_url"):
        if str(card.get(field) or "").strip():
            return True
    return False


def rescue_result_links(result: dict, *, progress=None) -> tuple[dict, list[str]]:
    """Rescue only the cards the UI will actually show; ranking stays untouched."""
    payload = dict(result)
    logs: list[str] = []
    attempted = resolved = primary = secondary = 0

    def rescue_cards(cards: list[dict], label: str) -> list[dict]:
        nonlocal attempted, resolved, primary, secondary
        output: list[dict] = []
        for index, raw in enumerate(cards, start=1):
            card = dict(raw)
            if _has_direct_link(card):
                output.append(card)
                continue

            company = str(card.get("company") or "").strip()
            title = str(card.get("title") or "").strip()
            if not company or not title:
                output.append(card)
                continue

            attempted += 1
            prefix = f"[{label} {index:02}] {company} — {title}"
            if progress:
                progress(prefix)
            link = rescue_exact_link(company, title)
            if link is None:
                line = "    -> no verified primary/secondary exact-role page found"
                logs.append(prefix + " | unresolved")
                if progress:
                    progress(line)
                output.append(card)
                continue

            resolved += 1
            if link.kind == "official_exact":
                primary += 1
                card["primary_source_url"] = link.url
                card["official_job_url"] = link.url
                card["application_url"] = card.get("application_url") or link.url
            else:
                secondary += 1
                card["secondary_source_url"] = link.url
                card["application_url"] = card.get("application_url") or link.url

            card["job_page_url"] = link.url
            card["job_page_kind"] = link.kind
            card["job_page_confidence"] = link.confidence
            card["search_resolution_status"] = "resolved_job_page"
            card["ui_link_rescue"] = {
                "query": link.query,
                "result_title": link.result_title,
            }
            logs.append(prefix + f" | {link.kind} | {link.url}")
            if progress:
                progress(f"    -> {link.kind}: {link.url}")
            output.append(card)
        return output

    payload["top_matches"] = rescue_cards(list(payload.get("top_matches") or []), "LINK RESCUE")
    payload["related_jobs"] = rescue_cards(list(payload.get("related_jobs") or []), "RELATED LINK")
    metrics = dict(payload.get("metrics") or {})
    metrics.update(
        {
            "ui_link_rescue_attempted": attempted,
            "ui_link_rescue_resolved": resolved,
            "ui_link_rescue_primary": primary,
            "ui_link_rescue_secondary": secondary,
        }
    )
    payload["metrics"] = metrics
    return payload, logs
