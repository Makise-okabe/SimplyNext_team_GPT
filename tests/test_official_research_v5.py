from datetime import datetime, timezone

from career_agent.job_identity.official_research import (
    _direct_candidates,
    _is_noise_url,
    build_provenance,
    evidence_tier,
    extract_job_id_from_url,
    focus_email_for_target,
    infer_initial_record_kind,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity


def _identity(**updates) -> JobIdentity:
    payload = {
        "source_message_id": "m1",
        "signal_index": 1,
        "company": "IBM",
        "title": "Associate Application Developer-AWS Cloud",
        "location": "Bangkok",
        "opportunity_type": "full_time",
        "direct_urls": [],
        "identity_strength": "moderate",
        "source_fingerprint": "fp",
    }
    payload.update(updates)
    return JobIdentity(**payload)


def test_company_careers_domain_is_official() -> None:
    identity = _identity()
    url = "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733"

    assert evidence_tier(identity, url) == "official"
    assert extract_job_id_from_url(url) == "88733"


def test_secondary_and_institutional_sources_are_not_official() -> None:
    identity = _identity()

    assert evidence_tier(identity, "https://www.linkedin.com/jobs/view/123") == "secondary"
    assert evidence_tier(identity, "https://careeraxis.ntu.edu.sg/Form.aspx?id=1") == "institutional"


def test_duckduckgo_ad_redirect_is_noise() -> None:
    assert _is_noise_url("https://duckduckgo.com/y.js?ad_domain=example.com") is True
    assert _is_noise_url("https://www.bing.com/aclick?ld=abc") is True


def test_direct_official_job_url_makes_concrete_job_posting() -> None:
    identity = _identity(
        direct_urls=[
            "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733"
        ]
    )

    assert infer_initial_record_kind(identity) == "job_posting"
    candidate = _direct_candidates(identity)[0]
    assert candidate.tier == "official"
    assert candidate.relation == "exact_posting"
    assert candidate.score >= 95


def test_programme_title_is_classified_separately() -> None:
    identity = _identity(
        company="Union Maritime Services",
        title="2026 Graduate Programme",
        opportunity_type="full_time",
    )

    assert infer_initial_record_kind(identity) == "programme"


def test_provenance_returns_original_outlook_pointer_not_raw_email() -> None:
    email = EmailMessage(
        message_id="m1",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Jobs",
        received_at=datetime(2026, 2, 8, tzinfo=timezone.utc),
        transport_sender_name="Du Yanzhang",
        transport_sender_email="example@u.nus.edu",
        links=[
            "https://outlook.live.com/owa/?ItemID=abc",
            "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733",
        ],
        attachments=["JD.pdf"],
    )

    provenance = build_provenance(email)

    assert provenance.original_email_url == "https://outlook.live.com/owa/?ItemID=abc"
    assert provenance.attachment_names == ["JD.pdf"]
    assert not hasattr(provenance, "body_text")


def test_focus_email_for_target_keeps_relevant_window_and_company_job_links() -> None:
    text = "noise " * 1500 + "IBM Associate Application Developer-AWS Cloud Bangkok" + " tail " * 1500
    email = EmailMessage(
        message_id="m1",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        body_text=text,
        links=[
            "https://outlook.live.com/owa/?ItemID=abc",
            "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733",
            "https://www.marvell.com/company/careers.html",
        ],
    )

    focused = focus_email_for_target(
        email,
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
    )

    assert "Associate Application Developer-AWS Cloud" in focused.body_text
    assert len(focused.body_text) < len(text)
    assert "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733" in focused.links
    assert not any("outlook.live.com" in url for url in focused.links)
