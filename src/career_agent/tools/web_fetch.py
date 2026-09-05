from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, quote

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
    headings: tuple[str, ...] = ()
    job_postings: tuple[dict, ...] = ()
    extraction_method: str = "html"
    link_labels: tuple[tuple[str, str], ...] = ()


def public_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not hostname or parsed.username or parsed.password:
            return False
        if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
            return False
        try:
            return ipaddress.ip_address(hostname).is_global
        except ValueError:
            return "." in hostname
    except ValueError:
        return False


def _postings(value):
    if isinstance(value, list):
        for item in value:
            yield from _postings(item)
    elif isinstance(value, dict):
        kinds = value.get("@type", [])
        if "JobPosting" in ([kinds] if isinstance(kinds, str) else kinds):
            yield value
        else:
            for item in value.values():
                if isinstance(item, (list, dict)):
                    yield from _postings(item)


def parse_html_page(url: str, final_url: str, status_code: int, content: str) -> FetchedPage:
    """Read JobPosting JSON-LD before removing scripts; never execute page code."""
    soup = BeautifulSoup(content, "html.parser")
    jobs: list[dict] = []
    fingerprints: set[str] = set()
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            for posting in _postings(json.loads(script.string or script.get_text())):
                key = json.dumps(posting, sort_keys=True)
                if key not in fingerprints:
                    fingerprints.add(key)
                    jobs.append(posting)
        except (ValueError, TypeError, RecursionError):
            continue
    links = tuple(dict.fromkeys(
        absolute for anchor in soup.find_all("a", href=True)
        if public_http_url(absolute := urljoin(final_url, str(anchor.get("href") or "")))
    ))[:500]
    headings = tuple(h.get_text(" ", strip=True) for h in soup.find_all("h1"))
    link_labels = tuple(
        (urljoin(final_url, str(a.get("href") or "")), a.get_text(" ", strip=True))
        for a in soup.find_all("a", href=True)
    )[:500]
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return FetchedPage(url, final_url, status_code, title, text[:30000], links, headings, tuple(jobs), link_labels=link_labels)


def _ats_posting(client, url: str) -> dict | None:
    """Read a public ATS detail resource for the *observed* job URL only."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host.endswith(".myworkdayjobs.com") and "job" in parts:
        index = parts.index("job")
        if index < 1:
            return None
        tenant, site = host.split(".")[0], parts[index - 1]
        path = "/".join(parts[index:])
        endpoint = f"https://{host}/wday/cxs/{quote(tenant, safe='')}/{quote(site, safe='')}/{path}"
        response = client.get(endpoint)
        response.raise_for_status()
        info = response.json().get("jobPostingInfo") or {}
        if not info.get("title") or not info.get("jobDescription"):
            return None
        return {
            "@type": "JobPosting", "title": info["title"],
            "description": info["jobDescription"], "identifier": info.get("jobReqId"),
            "jobLocation": info.get("location", ""), "employmentType": info.get("timeType", ""),
            "url": url,
        }
    if host in {"boards.greenhouse.io", "job-boards.greenhouse.io"} and len(parts) >= 3 and parts[1] == "jobs" and parts[2].isdigit():
        endpoint = f"https://boards-api.greenhouse.io/v1/boards/{quote(parts[0], safe='')}/jobs/{parts[2]}"
        response = client.get(endpoint)
        response.raise_for_status()
        info = response.json()
        if not info.get("title") or not info.get("content"):
            return None
        return {
            "@type": "JobPosting", "title": info["title"], "description": info["content"],
            "identifier": str(info.get("id", "")), "jobLocation": (info.get("location") or {}).get("name", ""),
            "url": url,
        }
    return None


def fetch_public_page(url: str, timeout_seconds: float = 12.0) -> FetchedPage:
    if not public_http_url(url):
        raise ValueError("Expected a public HTTP(S) URL")
    headers = {"User-Agent": "Mozilla/5.0 SimplyNext/0.2", "Accept-Language": "en-SG,en;q=0.9"}
    with httpx.Client(follow_redirects=False, timeout=timeout_seconds, headers=headers) as client:
        destination = url
        for _ in range(6):
            response = client.get(destination)
            if response.is_redirect:
                destination = urljoin(str(response.url), response.headers.get("location", ""))
                if not public_http_url(destination):
                    raise ValueError("Redirect is not a public HTTP(S) URL")
                continue
            break
        response.raise_for_status()
        final_url = str(response.url)
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type:
            return FetchedPage(url, final_url, response.status_code, "", "")
        page = parse_html_page(url, final_url, response.status_code, response.text)
        if not page.job_postings:
            try:
                posting = _ats_posting(client, final_url)
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                posting = None
            if posting:
                return FetchedPage(**{**page.__dict__, "job_postings": (posting,), "extraction_method": "public_ats_detail"})
        return page
