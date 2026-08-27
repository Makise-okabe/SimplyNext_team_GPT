from __future__ import annotations

import base64
from urllib.parse import urlparse, urlunparse

from career_agent.nodes.extract_signal import is_candidate_url
from career_agent.tools.web_fetch import fetch_public_page

MAX_PAGES_PER_EMAIL = 12


def decode_mailjet_tracking_url(url: str) -> str:
    """Decode Mailjet mjt.lu tracking links when the destination is embedded.

    Mailjet tracking URLs often end with a URL-safe base64 encoded destination.
    If decoding fails, return the original URL unchanged.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host.endswith("mjt.lu") or "/lnk/" not in parsed.path:
        return url

    encoded = parsed.path.rstrip("/").split("/")[-1]
    if not encoded:
        return url

    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
    except Exception:
        return url

    if not decoded.startswith(("http://", "https://")):
        return url

    destination = urlparse(decoded)
    if parsed.fragment and not destination.fragment:
        destination = destination._replace(fragment=parsed.fragment)

    return urlunparse(destination)


def resolve_links(state: dict) -> dict:
    """Follow public candidate URLs and capture redirect destination + page text."""
    signals = state.get("opportunity_signals") or []
    urls: list[str] = []
    seen: set[str] = set()

    for signal in signals:
        for url in signal.get("urls") or []:
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

    for original_url in urls:
        fetch_url = decode_mailjet_tracking_url(original_url)
        try:
            page = fetch_public_page(fetch_url)
            resolved_pages.append(
                {
                    "requested_url": original_url,
                    "decoded_url": fetch_url if fetch_url != original_url else None,
                    "final_url": page.final_url,
                    "status_code": page.status_code,
                    "title": page.title,
                    "text": page.text,
                }
            )
        except Exception as exc:
            errors.append(f"link resolution failed for {original_url}: {exc}")

    return {"resolved_pages": resolved_pages, "errors": errors}
