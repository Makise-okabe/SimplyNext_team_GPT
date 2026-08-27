from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from career_agent.graph.state import AgentState
from career_agent.models.email import EmailMessage
from career_agent.nodes.extract_signal import extract_signal
from career_agent.nodes.filter_email import filter_email
from career_agent.nodes.normalize_email import normalize_email
from career_agent.nodes.research_job import research_job
from career_agent.nodes.resolve_links import resolve_links
from career_agent.nodes.verify_job import verify_job


def normalize_email_node(state: dict) -> dict:
    raw = state.get("email") or {}
    message = raw if isinstance(raw, EmailMessage) else EmailMessage.model_validate(raw)
    normalized = normalize_email(message)
    return {
        "email": normalized.model_dump(mode="json"),
        "normalized_text": normalized.body_text,
        "extracted_links": normalized.links,
        "errors": state.get("errors", []),
    }


def _route_after_filter(state: dict) -> str:
    return "normalize_email" if state.get("is_career_email") else "end"


def build_workflow():
    """Compile Prototype Track B from one EmailMessage to verified Job records."""
    graph = StateGraph(AgentState)

    graph.add_node("filter_email", filter_email)
    graph.add_node("normalize_email", normalize_email_node)
    graph.add_node("extract_signal", extract_signal)
    graph.add_node("resolve_links", resolve_links)
    graph.add_node("research_job", research_job)
    graph.add_node("verify_job", verify_job)

    graph.add_edge(START, "filter_email")
    graph.add_conditional_edges(
        "filter_email",
        _route_after_filter,
        {
            "normalize_email": "normalize_email",
            "end": END,
        },
    )
    graph.add_edge("normalize_email", "extract_signal")
    graph.add_edge("extract_signal", "resolve_links")
    graph.add_edge("resolve_links", "research_job")
    graph.add_edge("research_job", "verify_job")
    graph.add_edge("verify_job", END)

    return graph.compile()


career_agent_workflow = build_workflow()
