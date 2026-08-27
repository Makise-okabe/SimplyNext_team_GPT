from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class ResearchedJob(BaseModel):
    company: str
    title: str
    location: str | None = None
    opportunity_type: str = "unknown"
    official_url: str | None = None
    deadline: str | None = None
    degree_requirements: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    visa_information: str | None = None
    raw_description: str = ""
    evidence: list[str] = Field(default_factory=list)


class ResearchedJobBatch(BaseModel):
    jobs: list[ResearchedJob] = Field(default_factory=list)


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    ).with_structured_output(ResearchedJobBatch)


def _page_for_signal(signal: dict, pages: list[dict]) -> dict | None:
    signal_urls = set(signal.get("urls", []))
    for page in pages:
        if page.get("requested_url") in signal_urls or page.get("final_url") in signal_urls:
            return page
    return None


def _research_one(signal: dict, page: dict | None) -> list[dict]:
    page_text = (page or {}).get("text", "")[:12000]
    page_title = (page or {}).get("title", "")
    final_url = (page or {}).get("final_url")

    prompt = f"""
You convert one NUS career opportunity signal plus optional public webpage evidence
into zero or more concrete JOB records.

Only return actual employment opportunities: internships or full-time/graduate jobs.
Do NOT return workshops, networking events, webinars, competitions, career fairs,
or generic career programmes unless they contain a specific job opening.
Do not invent facts. Empty fields are better than guesses.
Use opportunity_type only internship, full_time, or unknown.
If a public page is clearly the job posting, use its final URL as official_url.
Evidence must be short snippets grounded in the signal/page.

SIGNAL:
{signal}

PAGE TITLE:
{page_title}

FINAL URL:
{final_url}

PUBLIC PAGE TEXT:
{page_text}
""".strip()

    result = _build_llm().invoke(prompt)
    return [item.model_dump(mode="json") for item in result.jobs]


def research_job(state: dict) -> dict:
    """Turn opportunity signals into concrete candidate jobs using public evidence."""
    signals = state.get("opportunity_signals") or []
    pages = state.get("resolved_pages") or []
    errors = list(state.get("errors", []))
    candidate_jobs: list[dict] = []

    try:
        for signal in signals:
            if signal.get("opportunity_type") == "event":
                continue
            page = _page_for_signal(signal, pages)
            candidate_jobs.extend(_research_one(signal, page))
    except Exception as exc:
        errors.append(f"job research failed: {exc}")

    return {"candidate_jobs": candidate_jobs, "errors": errors}
