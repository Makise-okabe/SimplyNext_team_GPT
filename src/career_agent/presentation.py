"""The result boundary: only a verified destination can become a job button."""
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
