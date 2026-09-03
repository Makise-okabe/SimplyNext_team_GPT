from career_agent import related_job_discovery
from career_agent.related_job_discovery import discover_related_jobs
from career_agent.tools.web_search import SearchResult


def _student():
    return {
        "explicit_skills": ["machine learning", "python", "embedded systems", "software engineering"],
        "course_derived_skills": ["computer vision"],
    }


def test_related_discovery_prefers_existing_same_company_role_without_web(monkeypatch):
    web_calls = []

    def fake_search(query, **kwargs):
        web_calls.append(query)
        return []

    monkeypatch.setattr(related_job_discovery, "search_public_web", fake_search)

    existing_jobs = [
        {
            "source_key": "goh_ze_li",
            "source_message_id": "m1",
            "source_subject": "Career opportunities",
            "company": "Reolink",
            "title": "AI Engineer",
            "availability_status": "active_candidate",
            "opportunity_type": "full_time",
            "record_kind": "job_posting",
            "source_evidence": "Reolink | AI Engineer",
            "matching_evidence_level": "source_only",
            "matching_input_text": "Reolink AI Engineer machine learning Python",
        },
        {
            "source_key": "goh_ze_li",
            "source_message_id": "m1",
            "source_subject": "Career opportunities",
            "company": "Reolink",
            "title": "Backend Engineer",
            "availability_status": "active_candidate",
            "opportunity_type": "full_time",
            "record_kind": "job_posting",
            "source_evidence": "Reolink | Backend Engineer",
            "matching_evidence_level": "source_only",
            "matching_input_text": "Reolink Backend Engineer software engineering",
        },
    ]

    discovered, metrics = discover_related_jobs(
        top_rankings=[{"company": "Reolink", "title": "AI Engineer", "score": 90}],
        student_profile=_student(),
        existing_jobs=existing_jobs,
        max_companies=1,
        per_company=1,
    )

    assert len(discovered) == 1
    assert discovered[0].title == "Backend Engineer"
    assert discovered[0].research_basis == "related_role_from_existing_email_jobs"
    assert web_calls == []
    assert metrics.roles_discovered == 1
    assert metrics.companies_searched == 0


def test_related_discovery_rejects_home_product_and_non_job_pages(monkeypatch):
    def fake_search(query, **kwargs):
        return [
            SearchResult(
                title="HONOR",
                url="https://www.honor.com/global/",
                snippet="HONOR global website",
            ),
            SearchResult(
                title="Product - Goldilock",
                url="https://goldilock.com/product",
                snippet="Goldilock product page",
            ),
            SearchResult(
                title="AI Engineer - Example Co",
                url="https://example.com/careers/jobs/ai-engineer-123",
                snippet="Example Co AI Engineer careers role",
            ),
        ]

    monkeypatch.setattr(related_job_discovery, "search_public_web", fake_search)
    monkeypatch.setattr(
        related_job_discovery,
        "is_plausible_official_url",
        lambda url, company: "example.com" in url,
    )

    discovered, metrics = discover_related_jobs(
        top_rankings=[{"company": "Example Co", "title": "Data Scientist", "score": 90}],
        student_profile=_student(),
        existing_jobs=[],
        max_companies=1,
        per_company=2,
    )

    assert len(discovered) == 1
    assert discovered[0].title == "AI Engineer"
    assert "/jobs/" in discovered[0].job_page_url
    assert metrics.roles_discovered == 1
