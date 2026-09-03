from career_agent import job_link_resolver
from career_agent.job_link_resolver import resolve_job_link
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search import SearchResult


def _job(company="Reolink", title="AI Engineer"):
    return JobRecord(
        source_key="goh_ze_li",
        source_message_id="m1",
        source_subject="Career opportunities",
        company=company,
        title=title,
        opportunity_type="full_time",
        availability_status="active_candidate",
        record_kind="job_posting",
        source_evidence=f"{company} | {title}",
    )


def test_resolver_keeps_true_exact_secondary_page(monkeypatch):
    calls = []

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return [
            SearchResult(
                title="AI Engineer - Reolink",
                url="https://www.linkedin.com/jobs/view/123456",
                snippet="Reolink is hiring across AI and engineering teams",
            )
        ]

    monkeypatch.setattr(job_link_resolver, "search_public_web", fake_search)

    resolved, result = resolve_job_link(_job())
    assert result.url == "https://www.linkedin.com/jobs/view/123456"
    assert result.kind == "secondary_exact"
    assert resolved.secondary_source_url == result.url
    assert len(calls) == 1
    assert calls[0][0] == '"Reolink" "AI Engineer" careers job'


def test_resolver_prefers_official_exact(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="AI Engineer - Reolink",
                url="https://www.linkedin.com/jobs/view/123456",
                snippet="Reolink AI team",
            ),
            SearchResult(
                title="AI Engineer - Reolink Careers",
                url="https://reolink.com/careers/jobs/ai-engineer",
                snippet="Official Reolink careers page",
            ),
        ],
    )

    resolved, result = resolve_job_link(_job())
    assert result.kind == "official_exact"
    assert resolved.official_job_url == "https://reolink.com/careers/jobs/ai-engineer"
    assert resolved.job_page_confidence == "high"


def test_resolver_rejects_goldilock_full_stack_for_embedded_role(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="Full Stack Developer at Goldilock Secure",
                url="https://uk.linkedin.com/jobs/view/full-stack-developer-at-goldilock-secure-4366643587",
                snippet="Search results may mention Embedded Software Engineer in nearby text",
            )
        ],
    )
    resolved, result = resolve_job_link(
        _job("Goldilock", "Embedded Software Engineer (Aug - Nov/Dec 2026)")
    )
    assert result.url is None
    assert result.kind == "unresolved"
    assert resolved.job_page_url is None


def test_resolver_rejects_mobile_application_role_for_chip_design(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="Senior Mobile Application Engineer Jobs",
                url="https://ph.jobstreet.com/senior-mobile-application-engineer-jobs/in-Orchard-Central-Region-SG",
                snippet="Nanyang Singtech Chip Design Application Engineer",
            )
        ],
    )
    resolved, result = resolve_job_link(
        _job("Nanyang Singtech", "Chip Design / Application Engineer")
    )
    assert result.url is None
    assert resolved.job_page_url is None


def test_resolver_does_not_treat_facebook_as_face_ai(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="Facebook Artificial Intelligence Jobs",
                url="https://www.linkedin.com/jobs/facebook-artificial-intelligence-jobs",
                snippet="Face AI AI Research Engineer",
            )
        ],
    )
    resolved, result = resolve_job_link(_job("Face AI", "AI Research Engineer"))
    assert result.url is None
    assert resolved.job_page_url is None


def test_resolver_rejects_spirit_aerosystems_for_spirit_ai(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="College Internships | Spirit AeroSystems Careers",
                url="https://careers.spiritaero.com/intern-college",
                snippet="Spirit AI 2027 Fall Campus Recruitment",
            )
        ],
    )
    resolved, result = resolve_job_link(
        _job("Spirit AI", "Spirit AI 2027 Fall Campus Recruitment")
    )
    assert result.url is None
    assert result.kind == "unresolved"
    assert resolved.job_page_url is None


def test_resolver_rejects_probable_secondary_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="AI Engineering role - Reolink",
                url="https://www.linkedin.com/jobs/view/654321",
                snippet="Machine learning role at Reolink in Singapore",
            )
        ],
    )

    resolved, result = resolve_job_link(_job())
    assert result.url is None
    assert result.kind == "unresolved"
    assert resolved.search_resolution_status == "search_fallback_only"
