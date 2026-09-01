from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from career_agent.job_research_quality import (
    clean_jd_text,
    is_aggregator_url,
    is_plausible_official_url,
    page_is_closed,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_record import JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.web_fetch import FetchedPage, fetch_public_page
from career_agent.tools.web_search import SearchResult, search_public_web

MAX_SEARCH_RESULTS = 8
MAX_FETCH_CANDIDATES = 2
MIN_JD_CHARS = 500
MIN_JD_TITLE_OVERLAP = 0.20
MIN_SEARCH_TITLE_OVERLAP = 0.55
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
APPLICATION_HOSTS = {
    "forms.office.com",
    "forms.microsoft.com",
    "forms.cloud.microsoft",
    "forms.gle",
    "docs.google.com",
    "typeform.com",
    "www.typeform.com",
}
CAREER_URL_MARKERS = (
    "career",
    "careers",
    "/job/",
    "/jobs/",
    "jobdetail",
    "recruit",
    "workday",
    "greenhouse",
    "lever",
    "smartrecruiters",
    "successfactors",
    "taleo",
    "oraclecloud",
    "icims",
    "mokahr",
)

ProgressCallback = Callable[[str], None]


def _say(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _tokens(value: str | None) -> set[str]:
    return set(TOKEN_PATTERN.findall(_normalize(value)))


def _title_tokens(value: str | None) -> set[str]:
    stop = {"the", "and", "for", "with", "role", "position", "singapore"}
    return {
        token
        for token in TOKEN_PATTERN.findall(_normalize(value))
        if (len(token) >= 3 or token.isdigit()) and token not in stop
    }


def _title_overlap(title: str | None, text: str) -> float:
    source = _title_tokens(title)
    if not source:
        return 0.0
    return len(source & _tokens(text)) / len(source)


def _company_aliases(company: str | None) -> set[str]:
    raw = company or ""
    aliases: set[str] = set()
    for parenthetical in re.findall(r"\(([^)]{2,20})\)", raw):
        compact = re.sub(r"[^a-z0-9]", "", parenthetical.lower())
        if len(compact) >= 2:
            aliases.add(compact)
    legal_stop = {
        "the", "pte", "ltd", "limited", "inc", "private", "singapore",
        "company", "corporation", "corp", "solutions", "branch",
    }
    tokens = [token for token in TOKEN_PATTERN.findall(raw.lower()) if token not in legal_stop]
    aliases.update(token for token in tokens if len(token) >= 3)
    acronym_tokens = [token for token in tokens if token not in {"and", "of", "asia", "holdings"}]
    acronym = "".join(token[0] for token in acronym_tokens if token)
    if len(acronym) >= 2:
        aliases.add(acronym)
    compact = "".join(tokens)
    if len(compact) >= 4:
        aliases.add(compact)
    return aliases


def _company_match(company: str | None, text: str) -> bool:
    lowered = text.lower()
    compact_text = re.sub(r"[^a-z0-9]", "", lowered)
    for alias in _company_aliases(company):
        if len(alias) <= 3:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return True
        elif alias in lowered or alias in compact_text:
            return True
    return False


def _is_application_url(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in APPLICATION_HOSTS
    except ValueError:
        return False


def _is_concrete_job_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    value = f"{parsed.netloc.lower()} {parsed.path.lower()} {parsed.query.lower()} {parsed.fragment.lower()}"
    return any(
        marker in value
        for marker in (
            "/job/", "/jobs/", "jobdetail", "job-detail", "jobid=", "job_id=",
            "jobcode=", "requisitionid=", "reqid=", "/position/",
        )
    )


def _careerish_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(marker in lowered for marker in CAREER_URL_MARKERS)


def _result_text(result: SearchResult) -> str:
    return f"{result.title} {result.snippet} {result.url}"


def _page_matches(page: FetchedPage, signal: OpportunitySignal) -> bool:
    value = f"{page.title}\n{page.text}"
    return _title_overlap(signal.role_title, value) >= 0.20 and _company_match(signal.company, value)


def _usable_jd(page: FetchedPage, signal: OpportunitySignal) -> tuple[str | None, str | None]:
    if page_is_closed(page.text):
        return None, "closed"
    cleaned = clean_jd_text(page.text)
    if len(cleaned.strip()) < MIN_JD_CHARS:
        return None, "shell/insufficient JD text"
    if _title_overlap(signal.role_title, cleaned) < MIN_JD_TITLE_OVERLAP:
        return None, "JD title mismatch"
    return cleaned[:30_000], None


@dataclass
class ResearchContext:
    search_cache: dict[str, list[SearchResult]] = field(default_factory=dict)
    page_cache: dict[str, FetchedPage | None] = field(default_factory=dict)
    page_errors: dict[str, str] = field(default_factory=dict)
    search_calls: int = 0
    fetch_calls: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def search(self, query: str) -> list[SearchResult]:
        if query in self.search_cache:
            return self.search_cache[query]
        self.search_calls += 1
        try:
            results = search_public_web(query, max_results=MAX_SEARCH_RESULTS)
        except Exception as exc:
            results = []
            self.warnings.append(f"web search failed: {type(exc).__name__}: {exc} | query={query}")
        self.search_cache[query] = results
        return results

    def fetch(self, url: str) -> FetchedPage | None:
        if url in self.page_cache:
            return self.page_cache[url]
        self.fetch_calls += 1
        try:
            page = fetch_public_page(url, timeout_seconds=8.0)
        except Exception as exc:
            self.page_cache[url] = None
            self.page_errors[url] = f"{type(exc).__name__}: {exc}"
            self.warnings.append(f"page fetch failed: {url}: {type(exc).__name__}: {exc}")
            return None
        self.page_cache[url] = page
        return page


@dataclass
class RoleState:
    index: int
    signal: OpportunitySignal
    primary_url: str | None = None
    application_url: str | None = None
    jd_status: str = "unavailable"
    jd_source_url: str | None = None
    jd_text: str = ""
    availability_status: str = "active_candidate"
    research_status: str = "source_verified"
    research_confidence: str = "medium"
    research_basis: str = "trusted_nus_email"
    evidence_summary: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.jd_status == "fetched_official" or self.availability_status == "closed_by_official"


@dataclass
class CompanyResearchOutcome:
    job_records: list[JobRecord]
    search_calls: int
    fetch_calls: int
    warnings: list[str]
    errors: list[str]


def _application_url(signal: OpportunitySignal) -> str | None:
    return next((url for url in signal.urls if _is_application_url(url)), None)


def _source_official_urls(signal: OpportunitySignal) -> list[str]:
    values: list[str] = []
    for url in signal.urls:
        if _is_application_url(url) or is_aggregator_url(url):
            continue
        if is_plausible_official_url(url, signal.company):
            values.append(url)
    return list(dict.fromkeys(values))


def _try_official(state: RoleState, url: str, context: ResearchContext, *, direct: bool = False) -> bool:
    if state.finished:
        return True
    page = context.fetch(url)
    if page is None:
        return False

    page_matches = _page_matches(page, state.signal)
    if page_is_closed(page.text) and page_matches:
        state.primary_url = page.final_url or url
        state.availability_status = "closed_by_official"
        state.research_status = "verified_exact_job"
        state.research_confidence = "high"
        state.research_basis = "official_page_closed"
        state.evidence_summary.append("official employer/ATS page says the role is closed")
        return True

    cleaned, reason = _usable_jd(page, state.signal)
    if cleaned and page_matches:
        state.primary_url = page.final_url or url
        state.jd_status = "fetched_official"
        state.jd_source_url = page.final_url or url
        state.jd_text = cleaned
        state.research_status = "verified_exact_job"
        state.research_confidence = "high"
        state.research_basis = "official_company_or_ats_page"
        state.evidence_summary.append("official employer/ATS page matched the circulated role")
        return True

    final_url = page.final_url or url
    if direct and _is_concrete_job_url(final_url):
        state.primary_url = final_url
    state.warnings.append(f"official candidate did not yield usable JD: {url}: {reason or 'unverified'}")
    return False


def _rank_exact_role_candidates(state: RoleState, results: list[SearchResult]) -> list[SearchResult]:
    scored: list[tuple[float, SearchResult]] = []
    for result in results:
        if is_aggregator_url(result.url) or _is_application_url(result.url):
            continue
        value = _result_text(result)
        overlap = _title_overlap(state.signal.role_title, value)
        if overlap < MIN_SEARCH_TITLE_OVERLAP:
            continue
        if not _company_match(state.signal.company, value):
            continue
        plausible = is_plausible_official_url(result.url, state.signal.company)
        if not plausible and not _careerish_url(result.url):
            continue

        score = 100.0 * overlap
        if plausible:
            score += 25.0
        if _is_concrete_job_url(result.url):
            score += 20.0
        scored.append((score, result))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    ranked: list[SearchResult] = []
    for _, result in scored:
        if result.url in seen:
            continue
        seen.add(result.url)
        ranked.append(result)
    return ranked


def research_company_jobs(
    *,
    email: EmailMessage,
    source_key: str,
    company_items: list[tuple[int, OpportunitySignal]],
    context: ResearchContext,
    progress: ProgressCallback | None = None,
) -> CompanyResearchOutcome:
    """Fast deterministic Track B lookup.

    For each circulated role:
      1. Try an official URL already present in the trusted source.
      2. If still unresolved, run exactly one company + exact-title search.
      3. Fetch at most two official/career-like candidates.
      4. Stop. No LLM, no ATS discovery loop, no LinkedIn fallback.
    """
    before_search = context.search_calls
    before_fetch = context.fetch_calls
    before_warning = len(context.warnings)
    before_error = len(context.errors)

    states = [
        RoleState(
            index=index,
            signal=signal,
            availability_status="active_candidate" if signal.deadline_hint else "unknown",
            application_url=_application_url(signal),
            evidence_summary=["trusted NUS career source circulated this opportunity"],
        )
        for index, signal in company_items
    ]
    company = next((state.signal.company for state in states if state.signal.company), "<unknown>")
    _say(progress, f"  exact-title research: {company} ({len(states)} role(s))")

    for state in states:
        direct_urls = _source_official_urls(state.signal)
        if direct_urls:
            _say(progress, f"    source official: {state.signal.role_title}")
        for url in direct_urls[:1]:
            if _try_official(state, url, context, direct=True):
                break

        if state.finished:
            continue

        query = f'"{company}" "{state.signal.role_title}" careers job'
        _say(progress, f"    search exact title: {state.signal.role_title}")
        results = context.search(query)
        candidates = _rank_exact_role_candidates(state, results)
        _say(
            progress,
            f"      results={len(results)}, official candidates={min(len(candidates), MAX_FETCH_CANDIDATES)}",
        )
        for candidate in candidates[:MAX_FETCH_CANDIDATES]:
            if _try_official(state, candidate.url, context):
                break

    records: list[JobRecord] = []
    for state in states:
        signal = state.signal
        if not state.finished:
            state.research_status = "source_verified"
            state.research_confidence = "medium"
            state.research_basis = "trusted_nus_email_exact_title_unresolved"

        records.append(
            JobRecord(
                source_key=source_key,
                source_message_id=email.message_id,
                source_sender_email=email.sender_email,
                source_subject=email.subject,
                company=signal.company,
                title=signal.role_title,
                location=signal.location,
                opportunity_type=signal.opportunity_type,
                deadline_hint=signal.deadline_hint,
                availability_status=state.availability_status,
                target_major=signal.target_major,
                target_degree_level=signal.target_degree_level,
                source_urls=signal.urls,
                record_kind="job_posting",
                research_status=state.research_status,
                research_confidence=state.research_confidence,
                research_basis=state.research_basis,
                primary_source_url=state.primary_url,
                secondary_source_url=None,
                official_job_url=state.primary_url,
                application_url=state.application_url or state.primary_url or (signal.urls[0] if signal.urls else None),
                jd_status=state.jd_status,
                jd_source_url=state.jd_source_url,
                jd_text=state.jd_text,
                source_evidence=signal.raw_text,
                evidence_summary=state.evidence_summary,
                warnings=state.warnings,
                errors=state.errors,
            )
        )

    return CompanyResearchOutcome(
        job_records=records,
        search_calls=context.search_calls - before_search,
        fetch_calls=context.fetch_calls - before_fetch,
        warnings=context.warnings[before_warning:],
        errors=context.errors[before_error:],
    )
