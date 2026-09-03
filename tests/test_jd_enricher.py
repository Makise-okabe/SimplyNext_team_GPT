from career_agent import jd_enricher
from career_agent.jd_enricher import enrich_job_description
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_fetch import FetchedPage


def _job():
    return JobRecord(
        source_key="goh_ze_li",
        source_message_id="m1",
        source_subject="Career opportunities",
        company="Example Robotics",
        title="Embedded Engineer",
        opportunity_type="full_time",
        availability_status="active_candidate",
        record_kind="job_posting",
        source_evidence="Example Robotics | Embedded Engineer",
        job_page_url="https://example.com/jobs/embedded",
        job_page_kind="official_exact",
        job_page_confidence="high",
    )


def test_full_jd_is_kept_when_structured_page_is_available(monkeypatch):
    text = (
        "Example Robotics Embedded Engineer\nResponsibilities\n"
        + ("Develop embedded C firmware and debug microcontrollers. " * 30)
        + "\nRequirements\nDegree in electrical engineering and C/C++ experience."
    )
    monkeypatch.setattr(
        jd_enricher,
        "fetch_public_page",
        lambda url, timeout_seconds=10.0: FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="Embedded Engineer - Example Robotics",
            text=text,
        ),
    )
    enriched = enrich_job_description(_job())
    assert enriched.jd_status == "fetched_official"
    assert len(enriched.jd_text) >= 500


def test_partial_page_is_still_useful_evidence(monkeypatch):
    text = (
        "Example Robotics Embedded Engineer Singapore. "
        "Build embedded C firmware, test microcontrollers, debug hardware interfaces, "
        "and work with electrical engineers on product prototypes. "
        "Candidates should have electronics or computer engineering experience."
    )
    monkeypatch.setattr(
        jd_enricher,
        "fetch_public_page",
        lambda url, timeout_seconds=10.0: FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="Embedded Engineer - Example Robotics",
            text=text,
        ),
    )
    enriched = enrich_job_description(_job())
    assert enriched.jd_status == "partial_official"
    assert len(enriched.jd_text) >= 180
    assert enriched.job_page_url == _job().job_page_url
