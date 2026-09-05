"""Build result cards with verified pages and useful human-checkable fallbacks."""
from urllib.parse import quote_plus

from career_agent.job_page_verifier import clean_search_title
from career_agent.tools.web_fetch import public_http_url

LINK_FIELDS = ("job_page_url", "official_job_url", "application_url", "primary_source_url", "secondary_source_url")
SOURCE_LABELS = {"goh_ze_li": "Goh Ze Li · CDE Career Advisor", "talentconnect": "NUS TalentConnect", "web_discovered": "Discovered on the company website", "unknown": "Career source"}


def verified_job_url(card: dict) -> str | None:
    url = str(card.get("job_page_url") or "")
    if (card.get("link_verification_status") == "verified"
        and card.get("link_checked_at")
        and card.get("job_page_kind") in {"official_exact", "secondary_exact"}
        and card.get("availability_status") not in {"expired_by_source_deadline", "closed_by_official"}
        and public_http_url(url)):
        return url
    return None


def actionable_job_links(card: dict) -> list[tuple[str, str]]:
    """Return safe clickable next steps ordered from strongest to broadest."""
    verified = verified_job_url(card)
    if verified:
        label = "Open official job ↗" if card.get("job_page_kind") == "official_exact" else "Open secondary listing ↗"
        return [(label, verified)]

    links: list[tuple[str, str]] = []
    candidate = str(card.get("candidate_job_url") or "")
    if public_http_url(candidate):
        label = "Check likely official page ↗" if card.get("candidate_job_kind") == "official_candidate" else "Check secondary listing ↗"
        links.append((label, candidate))

    careers = str(card.get("company_careers_url") or "")
    if public_http_url(careers) and careers != candidate:
        links.append(("Open company careers ↗", careers))

    company = str(card.get("company") or "").strip()
    title = clean_search_title(str(card.get("title") or "").strip())
    location = str(card.get("location") or "Singapore").strip()
    if company and title:
        exact_query = quote_plus(f'"{company}" "{title}" official careers job')
        google_url = str(card.get("search_fallback_url") or f"https://www.google.com/search?q={exact_query}")
        if public_http_url(google_url):
            links.append(("Search official job ↗", google_url))
        linkedin_url = "https://www.linkedin.com/jobs/search/?" + f"keywords={quote_plus(company + ' ' + title)}&location={quote_plus(location)}"
        links.append(("Search LinkedIn ↗", linkedin_url))

    deduplicated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, url in links:
        if url not in seen:
            deduplicated.append((label, url))
            seen.add(url)
    return deduplicated


def job_card(job: dict, ranked: dict) -> dict:
    card = {**job, **ranked}
    # Ranking models cannot manufacture or restore source/link fields.
    for field in (*LINK_FIELDS, "job_page_kind", "job_page_confidence", "link_verification_status", "link_checked_at", "link_verification_reason", "availability_status"):
        card[field] = job.get(field)
    card["evidence_level"] = job.get("matching_evidence_level") or ranked.get("evidence_level") or "source_only"
    card["source_label"] = SOURCE_LABELS.get(str(job.get("source_key")), "Career source")
    if not verified_job_url(card):
        card.update({field: None for field in LINK_FIELDS})
        card["job_page_kind"] = "unresolved"
    else:
        # The verified job page is the next step. An old form is never an alias.
        card["application_url"] = card["job_page_url"]
    return card
