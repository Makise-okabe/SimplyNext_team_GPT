from __future__ import annotations

from career_agent.config import Settings


def should_extract_signal(state: dict) -> bool:
    """Return True only for the two career-email sources used in Prototype 1.

    Prototype 1 intentionally ingests only:
    - Goh Ze Li <zeli.goh@nus.edu.sg>
    - NUS TalentConnect <no-reply@kinobi.asia>

    Signal extraction itself will be implemented in M3. Keeping the source gate
    here prevents unrelated mailbox content from reaching the LLM later.
    """
    email = state.get("email") or {}
    sender = (email.get("sender_email") or "").strip().lower()
    return sender in Settings().trusted_senders


def extract_signal(state: dict) -> dict:
    """M3 placeholder for structured OpportunitySignal extraction.

    For now this node only enforces the Prototype 1 sender gate. In the next
    milestone, relevant email body text/links will be converted into one or
    more OpportunitySignal objects.
    """
    if not should_extract_signal(state):
        return {"opportunity_signals": []}

    return {
        "opportunity_signals": [],
        "errors": state.get("errors", []),
    }
