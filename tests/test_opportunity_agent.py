from career_agent import opportunity_agent
from career_agent.job_link_resolver import LinkResolution
from career_agent.models.job_record import JobRecord
from career_agent.opportunity_agent import run_opportunity_agent
from career_agent.related_job_discovery import RelatedDiscoveryMetrics


def _job(company: str, title: str) -> dict:
    return {
        "source_key": "goh_ze_li",
        "source_message_id": f"m-{company}",
        "source_subject": "Career opportunities",
        "company": company,
        "title": title,
        "location": "Singapore",
        "opportunity_type": "full_time",
        "availability_status": "active_candidate",
        "record_kind": "job_posting",
        "research_status": "source_verified",
        "research_confidence": "medium",
        "research_basis": "trusted_nus_email",
        "source_evidence": f"{company} | {title}",
    }


def _student():
    return {
        "explicit_skills": ["python", "machine learning", "embedded systems", "c/c++"],
        "course_derived_skills": ["digital design", "semiconductor"],
    }


def test_agent_ranks_first_then_resolves_only_selected_jobs(monkeypatch):
    jobs = [
        _job("AI Co", "AI Engineer"),
        _job("Embed Co", "Embedded Engineer"),
        _job("Marketing Co", "Marketing Intern"),
    ]
    resolved_companies = []

    def fake_resolve(job):
        resolved_companies.append(job.company)
        url = f"https://{job.company.lower().replace(' ', '')}.example/jobs/1"
        updated = job.model_copy(
            update={
                "job_page_url": url,
                "job_page_kind": "official_exact",
                "job_page_confidence": "high",
                "official_job_url": url,
                "application_url": url,
            }
        )
        return updated, LinkResolution(url, "official_exact", "high", "query", 1)

    monkeypatch.setattr(opportunity_agent, "resolve_job_link", fake_resolve)
    monkeypatch.setattr(opportunity_agent, "enrich_job_description", lambda job: job)
    monkeypatch.setattr(
        opportunity_agent,
        "discover_related_jobs",
        lambda **kwargs: ([], RelatedDiscoveryMetrics(0, 0, 0)),
    )

    result = run_opportunity_agent(
        student_profile=_student(),
        jobs=jobs,
        web_primary_count=2,
        web_exploration_count=0,
        semantic_shortlist_count=2,
    )

    assert len(resolved_companies) == 2
    assert "Marketing Co" not in resolved_companies
    assert result.metrics.links_resolved == 2
    assert result.metrics.web_selected == 2
    assert len(result.semantic_shortlist) == 2


def test_related_jobs_are_ranked_separately_from_email_jobs(monkeypatch):
    jobs = [_job("AI Co", "AI Engineer")]
    monkeypatch.setattr(
        opportunity_agent,
        "resolve_job_link",
        lambda job: (job, LinkResolution(None, "unresolved", "low", None, 0)),
    )
    monkeypatch.setattr(opportunity_agent, "enrich_job_description", lambda job: job)

    related = JobRecord(
        source_key="web_discovered",
        source_message_id="web:ai-co",
        source_subject="Related role discovered from company careers",
        company="AI Co",
        title="Computer Vision Engineer",
        availability_status="unknown",
        record_kind="job_posting",
        job_page_url="https://aico.example/jobs/cv",
        job_page_kind="official_probable",
        job_page_confidence="medium",
        official_job_url="https://aico.example/jobs/cv",
        application_url="https://aico.example/jobs/cv",
        source_evidence="AI Co Computer Vision Engineer",
    )
    monkeypatch.setattr(
        opportunity_agent,
        "discover_related_jobs",
        lambda **kwargs: ([related], RelatedDiscoveryMetrics(1, 4, 1)),
    )

    result = run_opportunity_agent(
        student_profile=_student(),
        jobs=jobs,
        web_primary_count=1,
        web_exploration_count=0,
    )
    assert result.metrics.related_jobs_discovered == 1
    assert len(result.related_jobs) == 1
    assert result.related_jobs[0]["source_key"] == "web_discovered"
    assert result.related_rankings[0]["company"] == "AI Co"


def test_parallel_research_keeps_rankings_and_counts(monkeypatch):
    from threading import Barrier
    from career_agent.research_session import current_session
    barrier = Barrier(3)
    jobs = [_job('AI Co', 'AI Engineer'), _job('Embed Co', 'Embedded Engineer'),
            _job('Chip Co', 'Semiconductor Engineer')]
    def resolve(job):
        barrier.wait(timeout=3)
        session = current_session()
        session.search_calls += 1
        session.fetch_calls += 2
        return job, LinkResolution(None, 'unresolved', 'low', None, 0)
    monkeypatch.setenv('SIMPLYNEXT_WEB_WORKERS', '3')
    monkeypatch.setattr(opportunity_agent, 'resolve_job_link', resolve)
    monkeypatch.setattr(opportunity_agent, 'enrich_job_description', lambda job: job)
    monkeypatch.setattr(opportunity_agent, 'discover_related_jobs',
                        lambda **kwargs: ([], RelatedDiscoveryMetrics(0, 0, 0)))
    result = run_opportunity_agent(student_profile=_student(), jobs=jobs,
                                   web_primary_count=3, web_exploration_count=0)
    assert result.metrics.search_calls == 3
    assert result.metrics.page_fetch_calls == 6
    assert [job['company'] for job in result.jobs] == [job['company'] for job in jobs]
    assert len(result.stage1_rankings) == 3
