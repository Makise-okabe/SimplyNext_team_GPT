from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, urlparse

import httpx

GREENHOUSE_API_ROOT = "https://boards-api.greenhouse.io/v1/boards"
GREENHOUSE_HOST_SUFFIX = "greenhouse.io"


@dataclass(frozen=True)
class GreenhouseJob:
    title: str
    url: str
    location: str = ""


def greenhouse_board_slug(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower()
    if not hostname.endswith(GREENHOUSE_HOST_SUFFIX):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    first = parts[0].strip()
    if first.lower() in {"jobs", "job", "embed"}:
        return None
    return first


def fetch_greenhouse_jobs(board_slug: str, timeout_seconds: float = 8.0) -> list[GreenhouseJob]:
    slug = (board_slug or "").strip()
    if not slug:
        return []
    response = httpx.get(
        f"{GREENHOUSE_API_ROOT}/{quote(slug, safe='')}/jobs",
        params={"content": "false"},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
        timeout=timeout_seconds,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()

    jobs: list[GreenhouseJob] = []
    seen: set[str] = set()
    for item in payload.get("jobs", []):
        title = " ".join(str(item.get("title") or "").split())
        url = str(item.get("absolute_url") or "").strip()
        location_obj = item.get("location") or {}
        location = " ".join(str(location_obj.get("name") or "").split())
        if not title or not url or url in seen:
            continue
        seen.add(url)
        jobs.append(GreenhouseJob(title=title, url=url, location=location))
    return jobs
