from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from career_agent.job_research_quality import (
    clean_jd_text,
    host,
    is_aggregator_url,
    is_plausible_official_url,
    page_is_closed,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_record import JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.greenhouse import fetch_greenhouse_jobs, greenhouse_board_slug
from career_agent.tools.web_fetch import FetchedPage, fetch_public_page
from career_agent.tools.web_search import SearchResult, search_public_web

ROLE_BATCH_SIZE = 4
MAX_SEARCH_RESULTS = 10
MAX_OFFICIAL_HOSTS = 2
MAX_SOURCE_LANDING_LINKS = 8
MAX_ROLE_FETCH_CANDIDATES = 3
MAX_LINKEDIN_FETCH_CANDIDATES = 2
MAX_GREENHOUSE_BOARDS = 2
MAX_GREENHOUSE_ROLE_CANDIDATES = 3
MIN_JD_CHARS = 500
MIN_JD_TITLE_OVERLAP = 0.20
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


def _is_linkedin_job_url(url: str | None) -> bool:
    value = host(url)
    if value != "linkedin.com" and not value.endswith(".linkedin.com"):
        return False
    try:
        return "/jobs" in urlparse(url or "").path.lower()
    except ValueError:
        return False


def _is_concrete_job_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    path = parsed.path.lower().rstrip("/")
    query = parsed.query.lower()
    fragment = parsed.fragment.lower()
    if any(marker in query for marker in ("jobid=", "job_id=", "jobcode=", "requisitionid=", "reqid=")):
        return True
    if any(marker in fragment for marker in ("/job/", "/jobs/", "jobid=", "jobcode=")):
        return True
    if any(marker in path for marker in ("/job/", "/jobdetail", "/job-detail", "/position/", "/career/", "/internship/")):
        return True
    jobs_index = path.find("/jobs/")
    return jobs_index >= 0 and len(path[jobs_index + len("/jobs/"):].strip("/")) >= 2


def _result_text(result: SearchResult) -> str:
    return f"{result.title} {result.snippet} {result.url}"


def _looks_official(signal: OpportunitySignal, result: SearchResult) -> bool:
    if is_aggregator_url(result.url) or _is_application_url(result.url):
        return False
    if is_plausible_official_url(result.url, signal.company):
        return True
    value = _result_text(result)
    if not _company_match(signal.company, value):
        return False
    try:
        parsed = urlparse(result.url)
    except ValueError:
        return False
    careerish = any(
        marker in f"{parsed.netloc.lower()} {parsed.path.lower()}"
        for marker in ("career", "jobs", "job", "recruit", "workday", "greenhouse", "lever", "position", "vacanc", "internship")
    )
    return careerish


def _official_score(signal: OpportunitySignal, result: SearchResult, *, preferred_hosts: set[str] | None = None) -> float:
    value = _result_text(result)
    score = 10.0 + 55.0 * _title_overlap(signal.role_title, value)
    if _is_concrete_job_url(result.url):
        score += 10.0
    if _company_match(signal.company, value):
        score += 15.0
    if is_plausible_official_url(result.url, signal.company):
        score += 15.0
    if preferred_hosts and host(result.url) in preferred_hosts:
        score += 15.0
    if signal.location and _normalize(signal.location) in _normalize(value):
        score += 5.0
    return min(100.0, score)


def _secondary_score(signal: OpportunitySignal, result: SearchResult) -> float:
    value = _result_text(result)
    score = 10.0 + 55.0 * _title_overlap(signal.role_title, value)
    if _company_match(signal.company, value):
        score += 15.0
    if _is_linkedin_job_url(result.url):
        score += 20.0
    if signal.location and _normalize(signal.location) in _normalize(value):
        score += 5.0
    return min(100.0, score)


def _page_matches(page: FetchedPage, signal: OpportunitySignal) -> bool:
    value = f"{page.title}\n{page.text}"
    return _title_overlap(signal.role_title, value) >= 0.15 and _company_match(signal.company, value)


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
        yield values[start:start + size]


def _role_or_query(items: list[tuple[int, OpportunitySignal]]) -> str:
    return " OR ".join(f'"{signal.role_title}"' for _, signal in items if signal.role_title)


def _location(items: list[tuple[int, OpportunitySignal]]) -> str:
    return next((signal.location for _, signal in items if signal.location), "Singapore")


@dataclass
class ResearchContext:
    search_cache: dict[str, list[SearchResult]] = field(default_factory=dict)
    page_cache: dict[str, FetchedPage | None] = field(default_factory=dict)
    greenhouse_cache: dict[str, list] = field(default_factory=dict)
    page_errors: dict[str, str] = field(default_factory=dict)
    search_calls: int = 0
    fetch_calls: int = 0
    greenhouse_calls: int = 0
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

    def greenhouse_jobs(self, board_slug: str):
        if board_slug in self.greenhouse_cache:
            return self.greenhouse_cache[board_slug]
        self.greenhouse_calls += 1
        try:
            jobs = fetch_greenhouse_jobs(board_slug, timeout_seconds=8.0)
        except Exception as exc:
            jobs = []
            self.warnings.append(
                f"Greenhouse board fetch failed: {board_slug}: {type(exc).__name__}: {exc}"
            )
        self.greenhouse_cache[board_slug] = jobs
        return jobs


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


def _direct_official_hosts(signal: OpportunitySignal) -> list[str]:
    values = []
    for url in signal.urls:
        if is_plausible_official_url(url, signal.company) and not _is_application_url(url):
            value = host(url)
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def _direct_official_jobs(signal: OpportunitySignal) -> list[str]:
    return [
        url for url in signal.urls
        if is_plausible_official_url(url, signal.company)
        and not _is_application_url(url)
        and _is_concrete_job_url(url)
    ]


def _direct_official_landings(signal: OpportunitySignal) -> list[str]:
    return [
        url for url in signal.urls
        if is_plausible_official_url(url, signal.company)
        and not _is_application_url(url)
        and not _is_concrete_job_url(url)
    ]


def _direct_secondary(signal: OpportunitySignal) -> list[str]:
    return [url for url in signal.urls if _is_linkedin_job_url(url)]


def _application_url(signal: OpportunitySignal) -> str | None:
    return next((url for url in signal.urls if _is_application_url(url)), None)


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
    if _is_concrete_job_url(final_url) and (direct or is_plausible_official_url(final_url, state.signal.company)):
        state.primary_url = final_url
    state.warnings.append(f"official candidate did not yield usable JD: {url}: {reason or 'unverified'}")
    return False


def _try_secondary(state: RoleState, url: str, context: ResearchContext) -> bool:
    if state.finished or not _is_linkedin_job_url(url):
        return state.finished
    page = context.fetch(url)
    if page is None:
        return False
    cleaned, reason = _usable_jd(page, state.signal)
    if cleaned and _page_matches(page, state.signal):
        state.secondary_url = page.final_url or url
        state.jd_status = "fetched_secondary"
        state.jd_source_url = page.final_url or url
        state.jd_text = cleaned
        state.research_status = "secondary_corroborated"
        state.research_confidence = "medium"
        state.research_basis = "official_primary_with_linkedin_jd" if state.primary_url else "linkedin_same_job_evidence"
        state.evidence_summary.append("LinkedIn job page matched the circulated role and supplied the JD")
        return True
    state.warnings.append(f"LinkedIn candidate did not yield usable JD: {url}: {reason or 'unverified'}")
    return False


def _rank_official_candidates(
    state: RoleState,
    results: list[SearchResult],
    *,
    preferred_hosts: set[str] | None = None,
) -> list[SearchResult]:
    scored: list[tuple[float, SearchResult]] = []
    for result in results:
        if is_aggregator_url(result.url) or _is_application_url(result.url):
            continue
        if preferred_hosts and host(result.url) not in preferred_hosts:
            continue
        value = _result_text(result)
        title_overlap = _title_overlap(state.signal.role_title, value)
        company_match = _company_match(state.signal.company, value)
        plausible = is_plausible_official_url(result.url, state.signal.company)
        if not plausible and not company_match and title_overlap < 0.35:
            continue
        scored.append((_official_score(state.signal, result, preferred_hosts=preferred_hosts), result))
    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    ranked: list[SearchResult] = []
    for _, result in scored:
        if result.url in seen:
            continue
        seen.add(result.url)
        ranked.append(result)
    return ranked


def _rank_linkedin_candidates(state: RoleState, results: list[SearchResult]) -> list[SearchResult]:
    scored: list[tuple[float, SearchResult]] = []
    for result in results:
        if not _is_linkedin_job_url(result.url):
            continue
        value = _result_text(result)
        if not _company_match(state.signal.company, value) and _title_overlap(state.signal.role_title, value) < 0.35:
            continue
        scored.append((_secondary_score(state.signal, result), result))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored]


def _candidate_links_from_landing(page: FetchedPage, state: RoleState) -> list[str]:
    scored: list[tuple[float, str]] = []
    page_host = host(page.final_url or page.requested_url)
    for url in page.links:
        if host(url) != page_host:
            continue
        value = url.replace("-", " ").replace("_", " ")
        overlap = _title_overlap(state.signal.role_title, value)
        if overlap <= 0:
            continue
        bonus = 0.25 if _is_concrete_job_url(url) else 0.0
        scored.append((overlap + bonus, url))
    scored.sort(reverse=True)
    return [url for _, url in scored[:MAX_SOURCE_LANDING_LINKS]]


def _fetch_ranked_official(
    states: list[RoleState],
    results: list[SearchResult],
    context: ResearchContext,
    progress: ProgressCallback | None,
    *,
    preferred_hosts: set[str] | None = None,
) -> None:
    for state in states:
        if state.finished:
            continue
        candidates = _rank_official_candidates(state, results, preferred_hosts=preferred_hosts)
        _say(progress, f"      {state.signal.role_title}: search results={len(results)}, fetch candidates={min(len(candidates), MAX_ROLE_FETCH_CANDIDATES)}")
        for candidate in candidates[:MAX_ROLE_FETCH_CANDIDATES]:
            if _try_official(state, candidate.url, context):
                break


def _greenhouse_board_candidates(company: str, results: list[SearchResult]) -> list[str]:
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for result in results:
        slug = greenhouse_board_slug(result.url)
        if not slug or slug in seen:
            continue
        value = _result_text(result)
        if not _company_match(company, value):
            continue
        score = 2 if "greenhouse" in result.url.lower() else 0
        score += 2 if _company_match(company, result.title) else 0
        scored.append((score, slug))
        seen.add(slug)
    scored.sort(reverse=True)
    return [slug for _, slug in scored[:MAX_GREENHOUSE_BOARDS]]


def _resolve_greenhouse(
    states: list[RoleState],
    company: str,
    context: ResearchContext,
    progress: ProgressCallback | None,
) -> None:
    unresolved = [state for state in states if not state.finished]
    if not unresolved:
        return

    discovery_results: list[SearchResult] = []
    for query in (
        f'"{company}" jobs greenhouse',
        f'"{company}" careers greenhouse',
    ):
        _say(progress, f"    Greenhouse discovery: {query}")
        discovery_results.extend(context.search(query))

    board_slugs = _greenhouse_board_candidates(company, discovery_results)
    _say(progress, f"      Greenhouse boards={len(board_slugs)}")
    for board_slug in board_slugs:
        jobs = context.greenhouse_jobs(board_slug)
        _say(progress, f"      Greenhouse board {board_slug}: jobs={len(jobs)}")
        for state in [item for item in states if not item.finished]:
            ranked: list[tuple[float, object]] = []
            for job in jobs:
                value = f"{job.title} {job.location}"
                overlap = _title_overlap(state.signal.role_title, value)
                if overlap < 0.50:
                    continue
                location_bonus = 0.1 if state.signal.location and _normalize(state.signal.location) in _normalize(job.location) else 0.0
                ranked.append((overlap + location_bonus, job))
            ranked.sort(key=lambda item: item[0], reverse=True)
            candidates = [job for _, job in ranked[:MAX_GREENHOUSE_ROLE_CANDIDATES]]
            _say(progress, f"      {state.signal.role_title}: Greenhouse candidates={len(candidates)}")
            for job in candidates:
                if _try_official(state, job.url, context, direct=True):
                    break


def research_company_jobs(*, email: EmailMessage, source_key: str, company_items: list[tuple[int, OpportunitySignal]], context: ResearchContext, progress: ProgressCallback | None = None) -> CompanyResearchOutcome:
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

    official_hosts: list[str] = []
    for state in states:
        official_hosts.extend(_direct_official_hosts(state.signal))
        direct_jobs = _direct_official_jobs(state.signal)
        if direct_jobs:
            _say(progress, f"    direct official/ATS: {state.signal.role_title}")
        for url in direct_jobs[:2]:
            if _try_official(state, url, context, direct=True):
                break

        if not state.finished:
            for landing_url in _direct_official_landings(state.signal)[:1]:
                _say(progress, f"    source official landing: {state.signal.role_title}")
                landing = context.fetch(landing_url)
                if landing is None:
                    continue
                for candidate_url in _candidate_links_from_landing(landing, state):
                    if _try_official(state, candidate_url, context):
                        break
                if state.finished:
                    break
    official_hosts = list(dict.fromkeys(official_hosts))[:MAX_OFFICIAL_HOSTS]

    unresolved = [state for state in states if not state.finished]
    if unresolved and not official_hosts:
        query = f'"{company}" careers jobs {_location(company_items)}'.strip()
        _say(progress, f"    discover official host: {query}")
        discovery = context.search(query)
        _say(progress, f"      discovery results={len(discovery)}")
        official_discovery = [
            result for result in discovery
            if any(_looks_official(state.signal, result) for state in unresolved)
        ]
        for result in official_discovery:
            value = host(result.url)
            if value:
                official_hosts.append(value)
        official_hosts = list(dict.fromkeys(official_hosts))[:MAX_OFFICIAL_HOSTS]
        exactish = [result for result in official_discovery if _is_concrete_job_url(result.url)]
        if exactish:
            _fetch_ranked_official(unresolved, exactish, context, progress)

    unresolved = [state for state in states if not state.finished]
    if unresolved and official_hosts:
        preferred_hosts = set(official_hosts)
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            batch_states = [state for state in states if state.index in {index for index, _ in role_batch}]
            results: list[SearchResult] = []
            for official_host in official_hosts:
                query = f"site:{official_host} {_role_or_query(role_batch)} {_location(role_batch)}".strip()
                _say(progress, f"    official batch search: {official_host} ({len(role_batch)} roles)")
                results.extend(context.search(query))
            _fetch_ranked_official(batch_states, results, context, progress, preferred_hosts=preferred_hosts)

        for state in [item for item in states if not item.finished]:
            for official_host in official_hosts:
                query = f'site:{official_host} "{state.signal.role_title}" {state.signal.location or "Singapore"}'
                _say(progress, f"    targeted official search: {state.signal.role_title}")
                results = context.search(query)
                _fetch_ranked_official([state], results, context, progress, preferred_hosts={official_host})
                if state.finished:
                    break

    # ATS fallback is company-bound and only resolves the roles circulated by the
    # trusted source; it never adds unrelated employer vacancies to the catalog.
    if any(not state.finished for state in states):
        _resolve_greenhouse(states, company, context, progress)

    for state in [item for item in states if not item.finished and not official_hosts]:
        query = f'"{company}" "{state.signal.role_title}" {state.signal.location or "Singapore"} careers job'
        _say(progress, f"    targeted official search: {state.signal.role_title}")
        results = context.search(query)
        _fetch_ranked_official([state], results, context, progress)

    for state in [item for item in states if not item.finished]:
        for url in _direct_secondary(state.signal)[:1]:
            if _try_secondary(state, url, context):
                break

    unresolved = [state for state in states if not state.finished]
    if unresolved:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            query = f'site:linkedin.com/jobs "{company}" {_role_or_query(role_batch)} {_location(role_batch)}'.strip()
            _say(progress, f"    LinkedIn fallback batch ({len(role_batch)} roles)")
            results = context.search(query)
            _say(progress, f"      LinkedIn search results={len(results)}")
            batch_states = [state for state in states if state.index in {index for index, _ in role_batch}]
            for state in batch_states:
                candidates = _rank_linkedin_candidates(state, results)
                _say(progress, f"      {state.signal.role_title}: LinkedIn fetch candidates={min(len(candidates), MAX_LINKEDIN_FETCH_CANDIDATES)}")
                for candidate in candidates[:MAX_LINKEDIN_FETCH_CANDIDATES]:
                    if _try_secondary(state, candidate.url, context):
                        break

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
