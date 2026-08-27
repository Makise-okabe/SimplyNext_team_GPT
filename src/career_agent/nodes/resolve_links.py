from __future__ import annotations

from career_agent.nodes.extract_signal import is_candidate_url
from career_agent.tools.web_fetch import fetch_public_page

MAX_PAGES_PER_EMAIL = 12


def resolve_links(state: dict) -> dict:
    """Follow public candidate URLs and capture redirect destination + page text."""
    signals = state.get("opportunity_signals") or []
    urls: list[str] = []
    seen: set[str] = set()

    for signal in signals:
        for url in signal.get("urls", []):
            if url in seen or not is_candidate_url(url):
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= MAX_PAGES_PER_EMAIL:
                break
        if len(urls) >= MAX_PAGES_PER_EMAIL:
            break

    resolved_pages: list[dict] = []
    errors = list(state.get("errors", []))

    for url in urls:
        try:
            page = fetch_public_page(url)
            resolved_pages.append(
                {
                    "requested_url": page.requested_url,
                    "final_url": page.final_url,
                    "status_code": page.status_code,
                    "title": page.title,
                    "text": page.text,
                }
            )
        except Exception as exc:
            errors.append(f"link resolution failed for {url}: {exc}")

    return {"resolved_pages": resolved_pages, "errors": errors}
