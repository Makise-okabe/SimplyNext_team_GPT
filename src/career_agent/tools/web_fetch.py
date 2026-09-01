from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class FetchedPage:
    requested_url: str
    final_url: str
    status_code: int
    title: str
    text: str
    links: tuple[str, ...] = ()


def fetch_public_page(url: str, timeout_seconds: float = 12.0) -> FetchedPage:
    """Fetch a public HTTP(S) page and return text plus public hyperlinks."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-SG,en;q=0.9",
    }

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout_seconds,
        headers=headers,
    ) as client:
        response = client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return FetchedPage(
            requested_url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            title="",
            text="",
            links=(),
        )

    soup = BeautifulSoup(response.text, "html.parser")
    final_url = str(response.url)
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        absolute = urljoin(final_url, href)
        if not absolute.startswith(("http://", "https://")) or absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
        if len(links) >= 500:
            break

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )

    return FetchedPage(
        requested_url=url,
        final_url=final_url,
        status_code=response.status_code,
        title=title,
        text=text[:30000],
        links=tuple(links),
    )
