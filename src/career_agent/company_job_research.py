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
MAX_SOURCE_LANDING_LINKS = 8
MIN_JD_CHARS = 500
MIN_JD_TITLE_OVERLAP = 0.20
MIN_OFFICIAL_SCORE = 45.0
MIN_SECONDARY_SCORE = 30.0
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
    if any(marker in path for marker in ("/job/", "/jobdetail", "/job-detail", "/position/", "/career/")):
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
        for marker in ("career", "jobs", "job", "recruit", "workday", "greenhouse", "lever", "position", "vacanc")
    )
    return careerish


def _official_score(signal: OpportunitySignal, result: SearchResult) -> float:
    value = _result_text(result)
    score = 20.0
    score += 55.0 * _title_overlap(signal.role_title, value)
    if _is_concrete_job_url(result.url):
        score += 10.0
    if _company_match(signal.company, value):
        score += 15.0
    if signal.location and _normalize(signal.location) in _normalize(value):
        score += 5.0
    return min(100.0, score)


def _secondary_score(signal: OpportunitySignal, result: SearchResult) -> float:
    value = _result_text(result)
    score = 10.0 + 55.0 * _title_overlap(signal.role_title, value)
    if _company_match(signal.company, value):
        score += 15.0
    if "linkedin.com" in host(result.url):
        score += 15.0
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
    return [url for url in signal.urls if is_secondary_url(url)]


def _application_url(signal: OpportunitySignal) -> str | None:
    return next((url for url in signal.urls if _is_application_url(url)), None)


def _try_official(state: RoleState, url: str, context: ResearchContext, *, direct: bool = False) -> bool:
    if state.finished:
        return True
    page = context.fetch(url)
    if page is None:
        return False
    page_matches = direct or _page_matches(page, state.signal)
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
    if _is_concrete_job_url(page.final_url or url):
        state.primary_url = page.final_url or url
    state.warnings.append(f"official candidate did not yield usable JD: {url}: {reason or 'unverified'}")
    return False


def _try_secondary(state: RoleState, url: str, context: ResearchContext) -> bool:
    if state.finished:
        return True
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
        state.research_basis = "official_primary_with_secondary_jd" if state.primary_url else "secondary_same_job_evidence"
        state.evidence_summary.append("secondary public job page matched the circulated role and supplied the JD")
        return True
    state.warnings.append(f"secondary candidate did not yield usable JD: {url}: {reason or 'unverified'}")
    return False


def _map_official(states: list[RoleState], results: list[SearchResult]) -> dict[int, list[tuple[float, SearchResult]]]:
    mapped = {state.index: [] for state in states}
    for result in results:
        for state in states:
            if state.finished or not _looks_official(state.signal, result):
                continue
            score = _official_score(state.signal, result)
            if score >= MIN_OFFICIAL_SCORE:
                mapped[state.index].append((score, result))
    for values in mapped.values():
        values.sort(key=lambda item: item[0], reverse=True)
    return mapped


def _map_secondary(states: list[RoleState], results: list[SearchResult]) -> dict[int, list[tuple[float, SearchResult]]]:
    mapped = {state.index: [] for state in states}
    for result in results:
        if not is_secondary_url(result.url):
            continue
        for state in states:
            if state.finished:
                continue
            score = _secondary_score(state.signal, result)
            if score >= MIN_SECONDARY_SCORE:
                mapped[state.index].append((score, result))
    for values in mapped.values():
        values.sort(key=lambda item: item[0], reverse=True)
    return mapped


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


def research_company_jobs(*, email: EmailMessage, source_key: str, company_items: list[tuple[int, OpportunitySignal]], context: ResearchContext, progress: ProgressCallback | None = None) -> CompanyResearchOutcome:
    before_search = context.search_calls
    before_fetch = context.fetch_calls
    before_warning = len(context.warnings)
    before_error = len(context.errors)

    states = [RoleState(index=index, signal=signal, availability_status="active_candidate" if signal.deadline_hint else "unknown", application_url=_application_url(signal), evidence_summary=["trusted NUS career source circulated this opportunity"]) for index, signal in company_items]
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
        official_discovery = [result for result in discovery if any(_looks_official(state.signal, result) for state in unresolved)]
        for result in official_discovery:
            value = host(result.url)
            if value:
                official_hosts.append(value)
        official_hosts = list(dict.fromkeys(official_hosts))[:MAX_OFFICIAL_HOSTS]
        mapped = _map_official(unresolved, official_discovery)
        for state in unresolved:
            for _, candidate in mapped.get(state.index, [])[:2]:
                if _try_official(state, candidate.url, context):
                    break

    unresolved = [state for state in states if not state.finished]
    if unresolved and official_hosts:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            batch_states = [state for state in states if state.index in {index for index, _ in role_batch}]
            results: list[SearchResult] = []
            for official_host in official_hosts:
                query = f"site:{official_host} {_role_or_query(role_batch)} {_location(role_batch)}".strip()
                _say(progress, f"    official batch search: {official_host} ({len(role_batch)} roles)")
                results.extend(context.search(query))
            mapped = _map_official(batch_states, results)
            for state in batch_states:
                for _, candidate in mapped.get(state.index, [])[:2]:
                    if _try_official(state, candidate.url, context):
                        break

        for state in [item for item in states if not item.finished]:
            for official_host in official_hosts:
                query = f'site:{official_host} "{state.signal.role_title}" {state.signal.location or "Singapore"}'
                _say(progress, f"    targeted official search: {state.signal.role_title}")
                candidates = _map_official([state], context.search(query)).get(state.index, [])
                for _, candidate in candidates[:2]:
                    if _try_official(state, candidate.url, context):
                        break
                if state.finished:
                    break

    unresolved = [state for state in states if not state.finished]
    if unresolved and not official_hosts:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            query = f'"{company}" careers {_role_or_query(role_batch)} {_location(role_batch)}'.strip()
            _say(progress, f"    official role batch search ({len(role_batch)} roles)")
            results = context.search(query)
            batch_states = [state for state in states if state.index in {index for index, _ in role_batch}]
            mapped = _map_official(batch_states, results)
            for state in batch_states:
                for _, candidate in mapped.get(state.index, [])[:2]:
                    if _try_official(state, candidate.url, context):
                        break

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
            batch_states = [state for state in states if state.index in {index for index, _ in role_batch}]
            mapped = _map_secondary(batch_states, results)
            for state in batch_states:
                for _, candidate in mapped.get(state.index, [])[:2]:
                    if _try_secondary(state, candidate.url, context):
                        break

    unresolved = [state for state in states if not state.finished]
    if unresolved:
        for role_batch in _chunks([(state.index, state.signal) for state in unresolved]):
            query = f'"{company}" {_role_or_query(role_batch)} {_location(role_batch)} Indeed JobStreet Glassdoor jobs'.strip()
            _say(progress, f"    broader secondary batch ({len(role_batch)} roles)")
            results = context.search(query)
            batch_states = [state for state in states if state.index in {index for index, _ in role_batch}]
            mapped = _map_secondary(batch_states, results)
            for state in batch_states:
                for _, candidate in mapped.get(state.index, [])[:2]:
                    if _try_secondary(state, candidate.url, context):
                        break

    records: list[JobRecord] = []
    for state in states:
        signal = state.signal
        if not state.finished:
            state.research_status = "source_verified"
            state.research_confidence = "medium"
            state.research_basis = "trusted_nus_email_web_unresolved"
        records.append(JobRecord(
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
        ))

    return CompanyResearchOutcome(
        job_records=records,
        search_calls=context.search_calls - before_search,
        fetch_calls=context.fetch_calls - before_fetch,
        warnings=context.warnings[before_warning:],
        errors=context.errors[before_error:],
    )
