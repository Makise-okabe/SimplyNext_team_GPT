"""Build result cards from exact role-page evidence only."""

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
    """Return direct role pages only; never manufacture a search-results URL."""
    links: list[tuple[str, str]] = []
    verified = verified_job_url(card)
    if verified:
        label = "Open official job ↗" if card.get("job_page_kind") == "official_exact" else "Open secondary listing ↗"
        if "linkedin.com" in verified.lower():
            label = "Open LinkedIn job ↗"
        links.append((label, verified))

        # A companion secondary URL is displayed only if this run separately
        # fetched and verified it. Ranking output cannot invent this evidence.
        secondary = str(card.get("secondary_source_url") or "")
        verified_attempt_urls = {
            str(attempt.get("final_url") or attempt.get("url") or "")
            for attempt in (card.get("link_attempts") or [])
            if attempt.get("status") == "verified"
        }
        if (card.get("job_page_kind") == "official_exact"
            and public_http_url(secondary)
            and secondary in verified_attempt_urls
            and secondary != verified):
            secondary_label = "Open LinkedIn job ↗" if "linkedin.com" in secondary.lower() else "Open secondary listing ↗"
            links.append((secondary_label, secondary))

    if not verified:
        candidate = str(card.get("candidate_job_url") or "")
        if public_http_url(candidate):
            kind = str(card.get("candidate_job_kind") or "")
            if kind == "official_archived":
                label = "Open archived official JD ↗"
            elif kind == "secondary_archived" and "linkedin.com" in candidate.lower():
                label = "Open archived LinkedIn JD ↗"
            elif kind == "secondary_archived":
                label = "Open archived secondary JD ↗"
            elif kind == "official_candidate":
                label = "Check possible official role ↗"
            elif "linkedin.com" in candidate.lower():
                label = "Check possible LinkedIn role ↗"
            else:
                label = "Check possible secondary role ↗"
            links.append((label, candidate))

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
    for field in (*LINK_FIELDS, "candidate_job_url", "candidate_job_kind", "candidate_job_reason", "company_careers_url", "job_page_kind", "job_page_confidence", "link_verification_status", "link_checked_at", "link_verification_reason", "availability_status"):
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
