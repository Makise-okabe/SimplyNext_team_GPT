from career_agent import shortlist_web_enrichment
from career_agent.company_job_research import CompanyResearchOutcome
from career_agent.shortlist_web_enrichment import enrich_stage1_shortlist


def _job(company: str, title: str, *, source_evidence: str = "email evidence") -> dict:
    return {
        "source_key": "goh_ze_li",
        "source_message_id": "m1",
        "source_sender_email": "career@example.edu",
        "source_subject": "Career opportunities",
        "company": company,
        "title": title,
        "location": "Singapore",
        "opportunity_type": "full_time",
        "availability_status": "active_candidate",
        "source_urls": [],
        "record_kind": "job_posting",
        "research_status": "source_verified",
        "research_confidence": "medium",
        "research_basis": "trusted_nus_email",
        "jd_status": "unavailable",
        "jd_text": "",
        "source_evidence": source_evidence,
        "matching_evidence_level": "source_only",
        "matching_input_text": source_evidence,
    }


def test_enrichment_only_researches_stage1_shortlist_and_upgrades_full_jd(monkeypatch):
    jobs = [
        _job("Chip Co", "Chip Design Engineer"),
        _job("Other Co", "Marketing Intern"),
    ]
    stage1 = [
        {"company": "Chip Co", "title": "Chip Design Engineer", "score": 90.0},
    ]
    calls = []

    def fake_research_company_jobs(*, email, source_key, company_items, context, progress=None):
        calls.append(company_items[0][1].company)
        signal = company_items[0][1]
        original = shortlist_web_enrichment._record(jobs[0])
        researched = original.model_copy(
            update={
                "availability_status": "active_candidate",
                "research_status": "verified_exact_job",
                "research_confidence": "high",
                "research_basis": "official_company_or_ats_page",
                "primary_source_url": "https://chip.example/jobs/123",
                "official_job_url": "https://chip.example/jobs/123",
                "application_url": "https://chip.example/jobs/123",
                "jd_status": "fetched_official",
                "jd_source_url": "https://chip.example/jobs/123",
                "jd_text": ("Chip Design Engineer semiconductor Cadence verification " * 20),
                "source_evidence": signal.raw_text,
                "evidence_summary": ["official employer/ATS page matched the circulated role"],
            }
        )
        return CompanyResearchOutcome(
            job_records=[researched],
            search_calls=1,
            fetch_calls=1,
            warnings=[],
            errors=[],
        )

    monkeypatch.setattr(
        shortlist_web_enrichment,
        "research_company_jobs",
        fake_research_company_jobs,
    )

    enriched, metrics = enrich_stage1_shortlist(
        all_jobs=jobs,
        stage1_rankings=stage1,
        stage1_top_n=1,
    )

    assert calls == ["Chip Co"]
    chip = next(item for item in enriched if item["company"] == "Chip Co")
    other = next(item for item in enriched if item["company"] == "Other Co")
    assert chip["matching_evidence_level"] == "full_jd"
    assert chip["official_job_url"] == "https://chip.example/jobs/123"
    assert other["matching_evidence_level"] == "source_only"
    assert metrics.selected == 1
    assert metrics.researched == 1
    assert metrics.upgraded_to_full_jd == 1
    assert metrics.still_source_only == 0
