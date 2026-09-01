from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

import httpx

from career_agent.job_research_quality import (
    clean_jd_text,
    host,
    is_aggregator_url,
    is_plausible_official_url,
    is_secondary_url,
    page_is_closed,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_record import JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.web_fetch import FetchedPage, fetch_public_page
from career_agent.tools.web_search import SearchResult, search_public_web

ROLE_BATCH_SIZE = 4
MAX_SEARCH_RESULTS = 10
MAX_OFFICIAL_HOSTS = 2
MIN_JD_CHARS = 500
MIN_JD_TITLE_OVERLAP = 0.20
MIN_OFFICIAL_RESULT_SCORE = 55.0
MIN_SECONDARY_RESULT_SCORE = 35.0
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
JOB_URL_MARKERS = (
    "/job/",
    "/jobs/",
    "/jobdetail",
    "/job-detail",
    "/position/",
    "/positions/",
    "/career/",
    "/careers/",
)
APPLICATION_HOSTS = {
    "forms.office.com",
    "forms.microsoft.com",
    "forms.cloud.microsoft",
    "forms.gle",
    "docs.google.com",
    "typeform.com",
    "www.typeform.com",
}

ProgressCallback = Callable[[str], None]


def _say(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _tokens(value: str | None) -> set[str]:
    return set(TOKEN_PATTERN.findall(_normalize(value)))


def _title_tokens(value: str | None) -> set[str]:
    stop = {"the", "and", "for", "with", "role", "position"}
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
        "the",
        "pte",
        "ltd",
        "limited",
        "inc",
        "private",
        "singapore",
        "company",
        "corporation",
        "corp",
    }
    raw_tokens = [
        token for token in TOKEN_PATTERN.findall(raw.lower()) if token not in legal_stop
    ]
    aliases.update(token for token in raw_tokens if len(token) >= 3)

    acronym_tokens = [
        token
        for token in raw_tokens
        if token not in {"and", "of", "asia", "holdings"}
    ]
    acronym = "".join(token[0] for token in acronym_tokens if token)
    if len(acronym) >= 2:
        aliases.add(acronym)

    compact = "".join(raw_tokens)
    if len(compact) >= 4:
        aliases.add(compact)
    return aliases


def _company_match(company: str | None, text: str) -> bool:
    compact_text = re.sub(r"[^a-z0-9]", "", text.lower())
    lowered = text.lower()
    for alias in _company_aliases(company):
        if len(alias) <= 3:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return True
        elif alias in compact_text or alias in lowered:
            return True
    return False


def _looks_job_like(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    path = parsed.path.lower()
    query = parsed.query.lower()
    return any(marker in path for marker in JOB_URL_MARKERS) or any(
        marker in query
        for marker in ("jobid=", "job_id=", "jobcode=", "requisitionid=", "reqid=")
    )


def _is_application_url(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in APPLICATION_HOSTS
    except ValueError:
        return False


def _result_text(result: SearchResult) -> str:
    return f"{result.title} {result.snippet} {result.url}"


def _looks_official_result(signal: OpportunitySignal, result: SearchResult) -> bool:
    if is_plausible_official_url(result.url, signal.company):
        return True
    if is_aggregator_url(result.url) or _is_application_url(result.url):
        return False

    value = _result_text(result)
    if not _company_match(signal.company, value):
        return False
    try:
        parsed = urlparse(result.url)
    except ValueError:
        return False
    careerish = any(
        marker in f"{parsed.netloc.lower()} {parsed.path.lower()}"
        for marker in ("career", "jobs", "job", "recruit", "workday", "greenhouse", "lever")
    )
    return careerish or _title_overlap(signal.role_title, value) >= 0.50


def _official_score(signal: OpportunitySignal, result: SearchResult) -> float:
    value = _result_text(result)
    score = 45.0
    score += 40.0 * _title_overlap(signal.role_title, value)
    if _looks_job_like(result.url):
        score += 10.0
    if _company_match(signal.company, value):
        score += 10.0
    if signal.location and _normalize(signal.location) in _normalize(value):
        score += 5.0
    return min(100.0, score)


def _secondary_score(signal: OpportunitySignal, result: SearchResult) -> float:
    value = _result_text(result)
    score = 15.0
    score += 50.0 * _title_overlap(signal.role_title, value)
    if _company_match(signal.company, value):
        score += 15.0
    result_host = host(result.url)
    if "linkedin.com" in result_host:
        score += 10.0
    if signal.location and _normalize(signal.location) in _normalize(value):
        score += 5.0
    return min(100.0, score)


def _page_company_title_match(page: FetchedPage, signal: OpportunitySignal) -> bool:
    value = f"{page.title}\n{page.text}"
    return _company_match(signal.company, value) and _title_overlap(signal.role_title, value) >= 0.15


def _usable_jd(page: FetchedPage, signal: OpportunitySignal) -> tuple[str | None, str | None]:
    if page_is_closed(page.text):
        return None, "closed"
    cleaned = clean_jd_text(page.text)
    if len(cleaned.strip()) < MIN_JD_CHARS:
        return None, "shell/insufficient JD text"
    if _title_overlap(signal.role_title, cleaned) < MIN_JD_TITLE_OVERLAP:
        return None, "JD title mismatch"
    return cleaned[:30_000], None


def _chunks(values: list, size: int = ROLE_BATCH_SIZE):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _role_or_query(items: list[tuple[int, OpportunitySignal]]) -> str:
    roles = [f'"{item.role_title}"' for _, item in items if item.role_title]
    return " OR ".join(roles)


def _company_location(items: list[tuple[int, OpportunitySignal]]) -> str:
    locations = [item.location for _, item in items if item.location]
    return locations[0] if locations else "Singapore"


@dataclass
class ResearchContext:
    search_cache: dict[str, list[SearchResult]] = field(default_factory=dict)
    page_cache: dict[str, FetchedPage | None] = field(default_factory=dict)
    page_errors: dict[str, str] = field(default_factory=dict)
    search_calls: int = 0
    fetch_calls: int = 0
    search_disabled_reason: str | None = None
    transient_search_failures: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def search(self, query: str) -> list[SearchResult]:
        if query in self.search_cache:
            return self.search_cache[query]
        if self.search_disabled_reason:
            return []

        self.search_calls += 1
        try:
            results = search_public_web(query, max_results=MAX_SEARCH_RESULTS)
        except Exception as exc:
            self.search_cache[query] = []
            warning = f"web search failed: {type(exc).__name__}: {exc} | query={query}"
            self.warnings.append(warning)
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403, 429}:
                self.search_disabled_reason = (
                    f"public search disabled for this run after HTTP {exc.response.status_code}"
                )
            elif isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
                self.transient_search_failures += 1
                if self.transient_search_failures >= 2:
                    self.search_disabled_reason = "public search disabled after repeated network failures"
            return []

        self.search_cache[query] = results
        self.transient_search_failures = 0
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
    secondary_url: str | None = None
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
        return self.jd_status in {"fetched_official", "fetched_secondary"} or self.availability_status == "closed_by_official"


@dataclass
class CompanyResearchOutcome:
    job_records: list[JobRecord]
    search_calls: int
    fetch_calls: int
    warnings: list[str]
    errors: list[str]


def _direct_official_urls(signal: OpportunitySignal) -> list[str]:
    return [
        url
        for url in signal.urls
        if is_plausible_official_url(url, signal.company) and not _is_application_url(url)
    ]


def _direct_secondary_urls(signal: OpportunitySignal) -> list[str]:
    return [url for url in signal.urls if is_secondary_url(url)]


def _application_url(signal: OpportunitySignal) -> str | None:
    return next((url for url in signal.urls if _is_application_url(url)), None)


def _accept_official_url(
    state: RoleState,
    url: str,
    context: ResearchContext,
    *,
    direct: bool = False,
) -> bool:
    if state.finished:
        return True
    state.primary_url = state.primary_url or url
    page = context.fetch(url)
    if page is None:
        return False

    if page_is_closed(page.text):
        if direct or _page_company_title_match(page, state.signal):
            state.availability_status = "closed_by_official"
            state.research_status = "verified_exact_job"
            state.research_confidence = "high"
            state.research_basis = "official_page_closed"
            state.evidence_summary.append("official employer/ATS page says the role is closed")
            return True
        return False

    cleaned, reason = _usable_jd(page, state.signal)
    if cleaned and (direct or _page_company_title_match(page, state.signal)):
        state.primary_url = page.final_url or url
        state.jd_status = "fetched_official"
        state.jd_source_url = page.final_url or url
        state.jd_text = cleaned
        state.research_status = "verified_exact_job"
        state.research_confidence = "high"
        state.research_basis = "official_company_or_ats_page"
        state.evidence_summary.append("official employer/ATS page matched the circulated role")
        return True

    if reason:
        state.warnings.append(f"official candidate did not yield usable JD: {url}: {reason}")
    return False


def _accept_secondary_url(
    state: RoleState,
    url: str,
    context: ResearchContext,
) -> bool:
    if state.finished:
        return True
    state.secondary_url = state.secondary_url or url
    page = context.fetch(url)
    if page is None:
        return False
    cleaned, reason = _usable_jd(page, state.signal)
    if cleaned:
        state.secondary_url = page.final_url or url
        state.jd_status = "fetched_secondary"
        state.jd_source_url = page.final_url or url
        state.jd_text = cleaned
        state.research_status = "secondary_corroborated"
        state.research_confidence = "medium"
        state.research_basis = (
            "official_primary_with_secondary_jd"
            if state.primary_url
            else "secondary_same_job_evidence"
        )
        state.evidence_summary.append("secondary public job page matched the circulated role and supplied the JD")
        return True
    if reason:
        state.warnings.append(f"secondary candidate did not yield usable JD: {url}: {reason}")
    return False


def _collect_official_hosts(states: list[RoleState]) -> list[str]:
    hosts: list[str] = []
    for state in states:
        for url in _direct_official_urls(state.signal):
            value = host(url)
            if value:
                hosts.append(value)
        if state.primary_url:
            value = host(state.primary_url)
            if value:
                hosts.append(value)
    return list(dict.fromkeys(hosts))[:MAX_OFFICIAL_HOSTS]


def _map_official_results(states: list[RoleState], results: list[SearchResult]) -> dict[int, list[tuple[float, SearchResult]]]:
    mapped: dict[int, list[tuple[float, SearchResult]]] = {state.index: [] for state in states}
    for result in results:
        for state in states:
            if state.finished or not _looks_official_result(state.signal, result):
                continue
            score = _official_score(state.signal, result)
            if score >= MIN_OFFICIAL_RESULT_SCORE:
                mapped[state.index].append((score, result))
    for values in mapped.values():
        values.sort(key=lambda item: item[0], reverse=True)
    return mapped


def _map_secondary_results(states: list[RoleState], results: list[SearchResult]) -> dict[int, list[tuple[float, SearchResult]]]:
    mapped: dict[int, list[tuple[float, SearchResult]]] = {state.index: [] for state in states}
    for result in results:
        if not is_secondary_url(result.url):
            continue
        for state in states:
            if state.finished:
                continue
            score = _secondary_score(state.signal, result)
            if score >= MIN_SECONDARY_RESULT_SCORE:
                mapped[state.index].append((score, result))
    for values in mapped.values():
        values.sort(key=lambda item: item[0], reverse=True)
    return mapped


def research_company_jobs(
    *,
    email: EmailMessage,
    source_key: str,
    company_items: list[tuple[int, OpportunitySignal]],
    context: ResearchContext,
    progress: ProgressCallback | None = None,
) -> CompanyResearchOutcome:
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
    _say(progress, f"  official-first research: {company} ({len(states)} role(s))")

    # Phase 0: source-provided official/ATS URLs. These cost zero searches and
    # establish reusable company hosts for every role in the same company.
    for state in states:
        direct_official = _direct_official_urls(state.signal)
        if direct_official:
            _say(progress, f"    direct official/ATS: {state.signal.role_title}")
        for url in direct_official[:2]:
            if _accept_official_url(state, url, context, direct=True):
                break

    official_hosts = _collect_official_hosts(states)
    unresolved = [state for state in states if not state.finished]

    # Phase 1: discover the employer/ATS host once per company when the source did
    # not already provide one. This is company-level work, not per-job work.
    if unresolved and not official_hosts and not context.search_disabled_reason:
        location = _company_location(company_items)
        query = f'"{company}" careers jobs {location}'.strip()
        _say(progress, f"    discover official host: {query}")
        discovery = context.search(query)
        official_discovery = [
            result
            for result in discovery
            if any(_looks_official_result(state.signal, result) for state in unresolved)
        ]
        for result in official_discovery:
            value = host(result.url)
            if value:
                official_hosts.append(value)
        official_hosts = list(dict.fromkeys(official_hosts))[:MAX_OFFICIAL_HOSTS]

        # A company-discovery query can already return concrete job pages, so map
        # those results before spending site-specific searches.
        mapped = _map_official_results(unresolved, official_discovery)
        for state in unresolved:
            candidates = mapped.get(state.index, [])
            if candidates:
                _accept_official_url(state, candidates[0][1].url, context)

    unresolved = [state for state in states if not state.finished]

    # Phase 2: search known official hosts in role batches. One query can resolve
    # several jobs from the same company. Only roles still unresolved after the
    # batch receive a targeted one-role site query.
    if unresolved and official_hosts and not context.search_disabled_reason:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            role_query = _role_or_query(role_batch)
            location = _company_location(role_batch)
            batch_states = [state for state in states if state.index in {i for i, _ in role_batch}]
            batch_results: list[SearchResult] = []
            for official_host in official_hosts:
                query = f"site:{official_host} {role_query} {location}".strip()
                _say(progress, f"    official batch search: {official_host} ({len(role_batch)} roles)")
                batch_results.extend(context.search(query))
            mapped = _map_official_results(batch_states, batch_results)
            for state in batch_states:
                candidates = mapped.get(state.index, [])
                if candidates:
                    _accept_official_url(state, candidates[0][1].url, context)

        for state in [item for item in states if not item.finished]:
            for official_host in official_hosts:
                query = f'site:{official_host} "{state.signal.role_title}" {state.signal.location or "Singapore"}'
                _say(progress, f"    targeted official search: {state.signal.role_title}")
                results = context.search(query)
                mapped = _map_official_results([state], results)
                candidates = mapped.get(state.index, [])
                if candidates and _accept_official_url(state, candidates[0][1].url, context):
                    break

    # If host discovery failed, do one company+role batch search rather than the
    # old exact+broad+fallback sequence for every individual job.
    unresolved = [state for state in states if not state.finished]
    if unresolved and not official_hosts and not context.search_disabled_reason:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            role_query = _role_or_query(role_batch)
            location = _company_location(role_batch)
            query = f'"{company}" careers {role_query} {location}'.strip()
            _say(progress, f"    official role batch search ({len(role_batch)} roles)")
            results = context.search(query)
            batch_states = [state for state in states if state.index in {i for i, _ in role_batch}]
            mapped = _map_official_results(batch_states, results)
            for state in batch_states:
                candidates = mapped.get(state.index, [])
                if candidates:
                    _accept_official_url(state, candidates[0][1].url, context)

    # Phase 3: secondary sources only for roles official research did not complete.
    # Direct secondary links are tried first, then LinkedIn in company batches,
    # then one broader secondary batch for the remaining roles.
    unresolved = [state for state in states if not state.finished]
    for state in unresolved:
        for url in _direct_secondary_urls(state.signal)[:1]:
            if _accept_secondary_url(state, url, context):
                break

    unresolved = [state for state in states if not state.finished]
    if unresolved and not context.search_disabled_reason:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            role_query = _role_or_query(role_batch)
            location = _company_location(role_batch)
            query = f'site:linkedin.com/jobs "{company}" {role_query} {location}'.strip()
            _say(progress, f"    LinkedIn fallback batch ({len(role_batch)} roles)")
            results = context.search(query)
            batch_states = [state for state in states if state.index in {i for i, _ in role_batch}]
            mapped = _map_secondary_results(batch_states, results)
            for state in batch_states:
                candidates = mapped.get(state.index, [])
                if candidates:
                    _accept_secondary_url(state, candidates[0][1].url, context)

    unresolved = [state for state in states if not state.finished]
    if unresolved and not context.search_disabled_reason:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            role_query = _role_or_query(role_batch)
            location = _company_location(role_batch)
            query = f'"{company}" {role_query} {location} Indeed JobStreet Glassdoor jobs'.strip()
            _say(progress, f"    broader secondary batch ({len(role_batch)} roles)")
            results = context.search(query)
            batch_states = [state for state in states if state.index in {i for i, _ in role_batch}]
            mapped = _map_secondary_results(batch_states, results)
            for state in batch_states:
                candidates = mapped.get(state.index, [])
                if candidates:
                    _accept_secondary_url(state, candidates[0][1].url, context)

    records: list[JobRecord] = []
    for state in states:
        signal = state.signal
        if not state.finished:
            state.research_status = "source_verified"
            state.research_confidence = "medium"
            state.research_basis = "trusted_nus_email_web_unresolved"
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
                secondary_source_url=state.secondary_url,
                official_job_url=state.primary_url,
                application_url=state.application_url or state.primary_url,
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
