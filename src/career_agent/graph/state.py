from typing import TypedDict


class AgentState(TypedDict, total=False):
    email: dict | None
    is_career_email: bool

    normalized_text: str
    extracted_links: list[str]

    opportunity_signals: list[dict]
    resolved_pages: list[dict]

    candidate_jobs: list[dict]
    verified_jobs: list[dict]

    unresolved_items: list[dict]
    errors: list[str]
