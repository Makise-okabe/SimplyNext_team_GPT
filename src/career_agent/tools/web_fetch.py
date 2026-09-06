from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, quote

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
    if host == "talent.lenovo.com.cn" and parsed.path.rstrip("/") == "/position/detail":
        job_id = (parse_qs(parsed.query).get("id") or [""])[0]
        if not job_id.isdigit():
            return None
        endpoint = "https://talent.lenovo.com.cn/gateway/jobBase/list?" + urlencode({"jobId": job_id})
        response = client.get(endpoint, headers={"Accept": "application/json", "portal-type": "PC"})
        response.raise_for_status()
        rows = ((response.json().get("result") or {}).get("rows") or [])
        info = rows[0] if rows and isinstance(rows[0], dict) else {}
        if not info.get("jobName"):
            return None
        active = bool(info.get("publishFlag")) and bool(info.get("activateFlag"))
        description = (
            "<h2>Responsibilities</h2>" + str(info.get("jobDuties") or "")
            + "<h2>Requirements</h2>" + str(info.get("jobRequirement") or "")
        )
        if not active:
            description += "<p>This job is no longer available.</p>"
        return {
            "@type": "JobPosting",
            "title": info["jobName"],
            "description": description,
            "hiringOrganization": {"name": "Lenovo"},
            "identifier": str(info.get("id") or job_id),
            "jobLocation": info.get("workPlace") or "China",
            "validThrough": None if active else "1970-01-01",
            "url": url,
        }
    if host in {"mycareersfuture.gov.sg", "www.mycareersfuture.gov.sg"} and "job" in parts:
        match = re.search(r"([0-9a-f]{32})$", parsed.path.rstrip("/"), re.I)
        if not match:
            return None
        job_id = match.group(1)
        response = client.get(f"https://api.mycareersfuture.gov.sg/v2/jobs/{job_id}")
        if response.status_code == 404:
            slug = parts[-1][:-33].replace("-", " ")
            return {
                "@type": "JobPosting",
                "title": slug,
                "description": "This job is no longer available. " * 4,
                "hiringOrganization": {"name": slug},
                "identifier": job_id,
                "validThrough": "1970-01-01",
                "url": url,
            }
        response.raise_for_status()
        info = response.json()
        company = (info.get("hiringCompany") or {}).get("name") or (info.get("postedCompany") or {}).get("name") or ""
        metadata = info.get("metadata") or {}
        dates = metadata.get("dates") or {}
        employment = [
            str(item.get("employmentType") or item.get("name") or "")
            for item in info.get("employmentTypes") or []
            if isinstance(item, dict)
        ]
        address = info.get("address") or {}
        location = {
            "address": {
                "addressLocality": address.get("street") or address.get("district") or "Singapore",
                "addressCountry": address.get("overseasCountry") or "Singapore",
            }
        }
        return {
            "@type": "JobPosting",
            "title": info.get("title") or "",
            "description": info.get("description") or "",
            "hiringOrganization": {"name": company},
            "identifier": info.get("uuid") or job_id,
            "jobLocation": location,
            "employmentType": employment,
            "validThrough": (
                dates.get("expiry")
                or dates.get("expiryDate")
                or metadata.get("expiryDate")
                or info.get("expiryDate")
            ),
            "url": url,
        }
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


def _dynamic_career_links(client, url: str) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Expose job-detail routes hidden behind supported client-rendered boards."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host != "talent.lenovo.com.cn" or parsed.path.rstrip("/") != "/position":
        return (), ()
    source_query = parse_qs(parsed.query)
    params = {"currentPage": "1", "pageSize": "100"}
    for key in ("projectType", "aiJobFlag", "jobTypeName", "workPlace", "deptId", "keyword"):
        value = (source_query.get(key) or [""])[0]
        if value:
            params[key] = value
    endpoint = "https://talent.lenovo.com.cn/gateway/jobBase/list?" + urlencode(params)
    response = client.get(endpoint, headers={"Accept": "application/json", "portal-type": "PC"})
    response.raise_for_status()
    rows = ((response.json().get("result") or {}).get("rows") or [])
    links: list[str] = []
    labels: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("id") or "").isdigit():
            continue
        if not row.get("publishFlag") or not row.get("activateFlag"):
            continue
        detail = f"https://talent.lenovo.com.cn/position/detail?id={row['id']}"
        links.append(detail)
        label = " ".join(filter(None, (
            str(row.get("jobName") or ""),
            str(row.get("typeName") or ""),
            str(row.get("firstDeptId") or ""),
        )))
        labels.append((detail, label))
    return tuple(dict.fromkeys(links)), tuple(labels)


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
        try:
            dynamic_links, dynamic_labels = _dynamic_career_links(client, final_url)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            dynamic_links, dynamic_labels = (), ()
        if dynamic_links:
            page = FetchedPage(**{
                **page.__dict__,
                "links": tuple(dict.fromkeys((*page.links, *dynamic_links))),
                "link_labels": tuple((*page.link_labels, *dynamic_labels)),
                "extraction_method": "public_career_api",
            })
        if not page.job_postings:
            try:
                posting = _ats_posting(client, final_url)
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                posting = None
            if posting:
                return FetchedPage(**{**page.__dict__, "job_postings": (posting,), "extraction_method": "public_ats_detail"})
        return page
