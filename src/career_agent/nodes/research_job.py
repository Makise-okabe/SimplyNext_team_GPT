from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from career_agent.tools.web_fetch import fetch_public_page
from career_agent.tools.web_search import search_public_web

LOGIN_WALL_HOSTS = {"nus-csm.symplicity.com"}
GENERIC_APPLICATION_HOSTS = {
    "forms.office.com",
    "forms.microsoft.com",
    "forms.cloud.microsoft",
    "forms.gle",
    "docs.google.com",
    "airtable.com",
    "www.airtable.com",
    "typeform.com",
    "www.typeform.com",
}
MAX_SEARCH_PAGES = 3
ALLOWED_OPPORTUNITY_TYPES = {"internship", "full_time", "unknown"}


class ResearchedJob(BaseModel):
    """Tolerant LLM schema before deterministic normalization.

    Every field that a model may reasonably leave blank accepts ``None``. This is
    deliberate: a missing optional fact should not turn a good extraction into a
    provider-level structured-output validation error.
    """

    company: str | None = None
    title: str | None = None
    location: str | None = None
    opportunity_type: str | None = None
    official_url: str | None = None
    application_url: str | None = None
    deadline: str | None = None
    degree_requirements: list[str] | None = None
    required_skills: list[str] | None = None
    preferred_skills: list[str] | None = None
    visa_information: str | None = None
    raw_description: str | None = None
    evidence: list[str] | None = None


class ResearchedJobBatch(BaseModel):
    jobs: list[ResearchedJob] = Field(default_factory=list)


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(ResearchedJobBatch)


def _page_for_signal(signal: dict, pages: list[dict]) -> dict | None:
    signal_urls = set(signal.get("urls", []))
    for page in pages:
        if page.get("requested_url") in signal_urls or page.get("final_url") in signal_urls:
            return page
    return None


def _page_is_useful(page: dict | None) -> bool:
    if not page or page.get("status_code") != 200:
        return False
    host = urlparse(page.get("final_url") or "").netloc.lower()
    return host not in LOGIN_WALL_HOSTS and bool(page.get("text"))


def _search_pages(signal: dict) -> list[dict]:
    company = signal.get("company") or ""
    title = signal.get("role_title") or ""
    if not company and not title:
        return []

    query = " ".join(part for part in (company, title, "careers jobs") if part).strip()
    pages: list[dict] = []

    for result in search_public_web(query, max_results=5):
        try:
            fetched = fetch_public_page(result.url)
        except Exception:
            continue

        pages.append(
            {
                "requested_url": result.url,
                "final_url": fetched.final_url,
                "status_code": fetched.status_code,
                "title": fetched.title or result.title,
                "text": fetched.text or result.snippet,
                "discovered_by_search": True,
            }
        )
        if len(pages) >= MAX_SEARCH_PAGES:
            break

    return pages


def _is_generic_application_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return False
    return host in GENERIC_APPLICATION_HOSTS


def normalize_researched_job(item: ResearchedJob, signal: dict) -> dict:
    """Convert tolerant LLM output into the strict downstream job payload."""
    payload = item.model_dump(mode="json")

    payload["company"] = payload.get("company") or signal.get("company")
    payload["title"] = payload.get("title") or signal.get("role_title")
    payload["location"] = payload.get("location") or signal.get("location")

    opportunity_type = payload.get("opportunity_type") or signal.get("opportunity_type") or "unknown"
    if opportunity_type not in ALLOWED_OPPORTUNITY_TYPES:
        opportunity_type = "unknown"
    payload["opportunity_type"] = opportunity_type

    # A generic form can be a perfectly useful application destination, but it
    # is not evidence of an employer-controlled careers page. Keep the link,
    # just classify it correctly.
    official_url = payload.get("official_url")
    application_url = payload.get("application_url")
    if _is_generic_application_url(official_url):
        application_url = application_url or official_url
        official_url = None

    payload["official_url"] = official_url
    payload["application_url"] = application_url
    payload["degree_requirements"] = payload.get("degree_requirements") or []
    payload["required_skills"] = payload.get("required_skills") or []
    payload["preferred_skills"] = payload.get("preferred_skills") or []
    payload["raw_description"] = payload.get("raw_description") or ""
    payload["evidence"] = payload.get("evidence") or []
    return payload


def _research_one(signal: dict, pages: list[dict]) -> list[dict]:
    evidence_blocks: list[str] = []
    for index, page in enumerate(pages, start=1):
        evidence_blocks.append(
            f"PAGE {index}\n"
            f"FINAL URL: {page.get('final_url')}\n"
            f"TITLE: {page.get('title', '')}\n"
            f"TEXT:\n{(page.get('text') or '')[:9000]}"
        )

    prompt = f"""
You convert one NUS career opportunity signal plus public-web evidence into zero or
more concrete JOB records.

Only return actual employment opportunities: internships or full-time/graduate jobs.
Do NOT return workshops, networking events, webinars, competitions, career fairs,
or generic career programmes unless they contain a specific job opening.
Prefer an official employer career/job posting over aggregators or school pages.
Do not invent facts. Null/empty fields are better than guesses.
Use opportunity_type only internship, full_time, or unknown.
Set official_url ONLY for an employer-controlled career/job page that describes the role.
If the useful link is a Microsoft Form, Google Form, Typeform, Airtable form or similar
application form, put it in application_url instead of official_url.
If the employer/company is not stated, return company as null rather than guessing.
raw_description may be null when no fuller description is supported.
Evidence must be short snippets grounded in the supplied signal/pages.

SIGNAL:
{signal}

PUBLIC WEB EVIDENCE:
{chr(10).join(evidence_blocks) if evidence_blocks else '<none>'}
""".strip()

    result = _build_llm().invoke(prompt)
    return [normalize_researched_job(item, signal) for item in result.jobs]


def research_job(state: dict) -> dict:
    """Turn opportunity signals into candidate jobs, searching publicly when needed."""
    signals = state.get("opportunity_signals") or []
    resolved_pages = list(state.get("resolved_pages") or [])
    errors = list(state.get("errors", []))
    candidate_jobs: list[dict] = []

    for signal in signals:
        if signal.get("opportunity_type") == "event":
            continue

        try:
            direct_page = _page_for_signal(signal, resolved_pages)
            evidence_pages = [direct_page] if _page_is_useful(direct_page) else []

            if not evidence_pages:
                searched_pages = _search_pages(signal)
                resolved_pages.extend(searched_pages)
                evidence_pages = searched_pages

            candidate_jobs.extend(_research_one(signal, evidence_pages))
        except Exception as exc:
            # Fail one signal, not the entire email. A newsletter can contain many
            # independent opportunities and one malformed model response should
            # not erase all of the others.
            label = signal.get("role_title") or signal.get("company") or "unknown opportunity"
            errors.append(f"job research failed for {label}: {exc}")

    return {
        "candidate_jobs": candidate_jobs,
        "resolved_pages": resolved_pages,
        "errors": errors,
    }
