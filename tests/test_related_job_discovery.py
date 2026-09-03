from career_agent import related_job_discovery
from career_agent.related_job_discovery import discover_related_jobs
from career_agent.tools.web_search import SearchResult


def _student():
    return {
        "explicit_skills": ["machine learning", "python", "embedded systems"],
        "course_derived_skills": ["computer vision"],
    }


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

    monkeypatch.setattr(related_job_discovery, "search_public_web_aggregated", fake_search)
    monkeypatch.setattr(
        related_job_discovery,
        "is_plausible_official_url",
        lambda url, company: "example.com" in url,
    )

    discovered, metrics = discover_related_jobs(
        top_rankings=[{"company": "Example Co", "title": "AI Engineer", "score": 90}],
        student_profile=_student(),
        existing_jobs=[],
        max_companies=1,
        per_company=2,
    )

    assert len(discovered) == 1
    assert discovered[0].title == "AI Engineer"
    assert "/jobs/" in discovered[0].job_page_url
    assert metrics.roles_discovered == 1
