from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from career_agent.job_research_quality import is_plausible_official_url, page_is_closed
from career_agent.tools.web_fetch import fetch_public_page
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
SEARCH_HOST_MARKERS = (
    "google.com", "bing.com", "duckduckgo.com",
)
APPLICATION_HOST_MARKERS = (
    "forms.office.com", "forms.microsoft.com", "forms.cloud.microsoft",
    "forms.gle", "docs.google.com", "typeform.com", "www.typeform.com",
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

    short = [token for token in tokens if len(token) <= 3]
    if len(tokens) >= 2 and short and all(token in identity_tokens for token in short):
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


def _is_search_url(url: str) -> bool:
    host = _host(url)
    return any(marker in host for marker in SEARCH_HOST_MARKERS)


def _is_ats(url: str) -> bool:
    host = _host(url)
    return any(marker in host for marker in ATS_HOST_MARKERS)


def _is_secondary(url: str) -> bool:
    host = _host(url)
    return any(marker in host for marker in SECONDARY_HOST_MARKERS)


def _is_application_host(url: str) -> bool:
    host = _host(url)
    return any(marker in host for marker in APPLICATION_HOST_MARKERS)


def _is_concrete_job_url(url: str) -> bool:
    value = (url or "").lower()
    return any(marker in value for marker in CONCRETE_JOB_MARKERS)


def _candidate_kind(company: str, url: str) -> str | None:
    if is_plausible_official_url(url, company) or _is_ats(url):
        return "official_exact"
    if _is_secondary(url):
        return "secondary_exact"
    return None


def _score(company: str, title: str, result: SearchResult) -> tuple[float, str, str] | None:
    identity = f"{result.title} {result.snippet} {result.url}"
    if not _company_match(company, identity):
        return None

    overlap = _title_overlap(title, identity)
    if overlap < 0.72:
        return None

    kind = _candidate_kind(company, result.url)
    concrete = _is_concrete_job_url(result.url)
    if kind == "official_exact" and concrete:
        return 200.0 + 100.0 * overlap, kind, "high"
    if kind == "secondary_exact" and concrete and overlap >= 0.82:
        return 120.0 + 100.0 * overlap, kind, "medium"
    return None


def _verify_page(company: str, title: str, url: str, kind: str) -> tuple[str | None, str]:
    """Fetch and verify a concrete exact-role page before it is ever displayed."""
    if not url or _is_search_url(url):
        return None, "search URL is not a job page"

    try:
        page = fetch_public_page(url, timeout_seconds=7.0)
    except Exception as exc:
        return None, f"fetch failed: {type(exc).__name__}"

    final_url = str(page.final_url or url).strip()
    if not final_url or _is_search_url(final_url):
        return None, "redirected to a search page"
    if page.status_code < 200 or page.status_code >= 400:
        return None, f"HTTP {page.status_code}"
    if not _is_concrete_job_url(final_url):
        return None, "redirected to a generic/non-job page"

    # Search snippets are only candidate discovery. The fetched page itself must
    # independently identify the role. Use page title + bounded body text + URL.
    identity = f"{page.title}\n{page.text[:12000]}\n{final_url}"
    overlap = _title_overlap(title, identity)
    if overlap < 0.66:
        return None, f"role title mismatch ({overlap:.2f})"

    official = kind == "official_exact"
    if official:
        # Employer/ATS URL identity is enough when the page body omits a legal
        # company name; otherwise require the company to appear in page content.
        if not (is_plausible_official_url(final_url, company) or _company_match(company, identity)):
            return None, "company mismatch"
    else:
        if not _is_secondary(final_url):
            return None, "secondary page redirected outside trusted hosts"
        if not _company_match(company, identity):
            return None, "company mismatch"
        if overlap < 0.78:
            return None, f"secondary title match too weak ({overlap:.2f})"

    # Closed pages are useful evidence but not a valid CTA for an active job card.
    if page_is_closed(page.text):
        return None, "page says role is closed/expired"

    return final_url, "verified"


def _application_reachable(url: str | None) -> str | None:
    value = str(url or "").strip()
    if not value or _is_search_url(value):
        return None
    # If application_url is itself a concrete job page, exact-role validation is
    # handled separately. This helper is only for trusted application-form hosts.
    if not _is_application_host(value):
        return None
    try:
        page = fetch_public_page(value, timeout_seconds=6.0)
    except Exception:
        return None
    if 200 <= page.status_code < 400 and page.final_url and not _is_search_url(page.final_url):
        return str(page.final_url)
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
    """Search, fetch, and verify a bounded exact-role primary/secondary page."""
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

        # Search-result score is not enough to stop anymore: page-level
        # verification below decides whether a result is actually usable.

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    # Bound page fetches so one UI card cannot explode runtime.
    for _, result, kind, confidence, query in scored[:6]:
        verified_url, _reason = _verify_page(company, title, result.url, kind)
        if not verified_url:
            continue
        return RescuedLink(
            url=verified_url,
            kind=kind,
            confidence=confidence,
            query=query,
            result_title=result.title,
        )
    return None


def _existing_exact_candidates(card: dict, company: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(url: str | None, kind: str | None = None) -> None:
        value = str(url or "").strip()
        if not value or value in seen or _is_search_url(value):
            return
        resolved_kind = kind or _candidate_kind(company, value)
        if resolved_kind not in {"official_exact", "secondary_exact"}:
            return
        seen.add(value)
        values.append((value, resolved_kind))

    page_kind = str(card.get("job_page_kind") or "").lower()
    if page_kind in {"official_exact", "secondary_exact"}:
        add(card.get("job_page_url"), page_kind)
    add(card.get("official_job_url"), "official_exact")
    add(card.get("primary_source_url"), "official_exact")
    add(card.get("secondary_source_url"), "secondary_exact")
    return values


def _clear_direct_fields(card: dict) -> None:
    for field in (
        "job_page_url", "official_job_url", "primary_source_url", "secondary_source_url",
    ):
        card[field] = None
    card["job_page_kind"] = "unresolved"
    card["job_page_confidence"] = "low"
    card["search_resolution_status"] = "not_searched"


def rescue_result_links(result: dict, *, progress=None) -> tuple[dict, list[str]]:
    """Verify every displayed link, reject bad ones, then rescue exact replacements."""
    payload = dict(result)
    logs: list[str] = []
    attempted = resolved = primary = secondary = 0
    existing_checked = existing_rejected = existing_verified = 0

    def rescue_cards(cards: list[dict], label: str) -> list[dict]:
        nonlocal attempted, resolved, primary, secondary
        nonlocal existing_checked, existing_rejected, existing_verified
        output: list[dict] = []

        for index, raw in enumerate(cards, start=1):
            card = dict(raw)
            company = str(card.get("company") or "").strip()
            title = str(card.get("title") or "").strip()
            if not company or not title:
                output.append(card)
                continue

            prefix = f"[{label} {index:02}] {company} — {title}"
            original_application = card.get("application_url")
            candidates = _existing_exact_candidates(card, company)
            _clear_direct_fields(card)

            verified: RescuedLink | None = None
            for url, kind in candidates:
                existing_checked += 1
                verified_url, reason = _verify_page(company, title, url, kind)
                if not verified_url:
                    existing_rejected += 1
                    logs.append(prefix + f" | rejected existing {kind} | {reason} | {url}")
                    if progress:
                        progress(f"{prefix}\n    -> rejected existing link: {reason}")
                    continue
                existing_verified += 1
                verified = RescuedLink(
                    url=verified_url,
                    kind=kind,
                    confidence="high" if kind == "official_exact" else "medium",
                    query="existing_backend_or_source_link",
                    result_title="verified fetched page",
                )
                break

            if verified is None:
                attempted += 1
                if progress:
                    progress(prefix + "\n    -> searching for a verified exact primary/secondary page...")
                verified = rescue_exact_link(company, title)

            if verified is None:
                card["application_url"] = _application_reachable(original_application)
                logs.append(prefix + " | unresolved after page verification")
                if progress:
                    progress("    -> no live exact-role page passed verification")
                output.append(card)
                continue

            resolved += 1
            if verified.kind == "official_exact":
                primary += 1
                card["primary_source_url"] = verified.url
                card["official_job_url"] = verified.url
            else:
                secondary += 1
                card["secondary_source_url"] = verified.url

            valid_application = _application_reachable(original_application)
            card["application_url"] = valid_application or verified.url
            card["job_page_url"] = verified.url
            card["job_page_kind"] = verified.kind
            card["job_page_confidence"] = verified.confidence
            card["search_resolution_status"] = "resolved_job_page"
            card["ui_link_verification"] = {
                "query": verified.query,
                "result_title": verified.result_title,
                "page_verified": True,
            }
            logs.append(prefix + f" | verified {verified.kind} | {verified.url}")
            if progress:
                progress(f"    -> verified {verified.kind}: {verified.url}")
            output.append(card)
        return output

    payload["top_matches"] = rescue_cards(list(payload.get("top_matches") or []), "LINK VERIFY")
    payload["related_jobs"] = rescue_cards(list(payload.get("related_jobs") or []), "RELATED VERIFY")

    metrics = dict(payload.get("metrics") or {})
    metrics.update(
        {
            "ui_link_existing_checked": existing_checked,
            "ui_link_existing_verified": existing_verified,
            "ui_link_existing_rejected": existing_rejected,
            "ui_link_rescue_attempted": attempted,
            "ui_link_rescue_resolved": resolved,
            "ui_link_rescue_primary": primary,
            "ui_link_rescue_secondary": secondary,
        }
    )
    payload["metrics"] = metrics
    return payload, logs
