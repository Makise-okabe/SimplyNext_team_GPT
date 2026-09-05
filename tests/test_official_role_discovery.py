from career_agent import job_link_resolver as resolver
from career_agent.models.job_record import JobRecord
from career_agent.presentation import actionable_job_links, job_card
from career_agent.tools.web_fetch import parse_html_page
from career_agent.tools.web_search import SearchResult


def job():
    return JobRecord(source_message_id="test", source_subject="Career opportunities", company="Reolink", title="AI Engineer (Aug - Nov/Dec 2026)")


def page(url, title="AI Engineer"):
    return parse_html_page(url, url, 200, f"<title>{title} - Reolink</title><h1>{title}</h1><p>Responsibilities: " + "Build machine learning systems. " * 30 + "</p>")


def test_official_found_after_linkedin(monkeypatch):
    calls = []
    official = "https://reolink.com/jobs/123"
    def search(query, **kwargs):
        calls.append(query)
        url = "https://www.linkedin.com/jobs/view/456" if len(calls) == 1 else official
        return [SearchResult("AI Engineer - Reolink", url, "")]
    monkeypatch.setattr(resolver, "search_public_web", search)
    monkeypatch.setattr(resolver, "fetch_public_page", lambda url, **kw: page(url))
    resolved, _ = resolver.resolve_job_link(job())
    assert resolved.job_page_url == official
    assert resolved.job_page_kind == "official_exact"
    assert resolved.secondary_source_url == "https://www.linkedin.com/jobs/view/456"
    links = actionable_job_links(job_card(resolved.model_dump(), {}))
    assert links == [
        ("Open official job ↗", official),
        ("Open LinkedIn job ↗", "https://www.linkedin.com/jobs/view/456"),
    ]
    assert len(calls) == 2


def test_careers_anchor_text_discovers_numeric_role(monkeypatch):
    landing = "https://reolink.com/join-us/"
    official = "https://reolink.com/jobs/123"
    def fetch(url, **kw):
        if url == landing:
            return parse_html_page(url, url, 200, '<title>Careers - Reolink</title><a href="/jobs/123">AI Engineer</a>')
        return page(url)
    monkeypatch.setattr(resolver, "search_public_web", lambda *a, **kw: [SearchResult("Careers - Reolink", landing, "")])
    monkeypatch.setattr(resolver, "fetch_public_page", fetch)
    resolved, _ = resolver.resolve_job_link(job())
    assert resolved.job_page_url == official


def test_known_wrong_role_is_not_retained_as_candidate(monkeypatch):
    url = "https://reolink.com/jobs/123"
    monkeypatch.setattr(resolver, "search_public_web", lambda *a, **kw: [SearchResult("AI Engineer - Reolink", url, "")])
    monkeypatch.setattr(resolver, "fetch_public_page", lambda url, **kw: page(url, "Mechanical Engineer"))
    resolved, _ = resolver.resolve_job_link(job())
    assert resolved.candidate_job_url is None
    assert resolved.job_page_url is None


def test_ranking_cannot_inject_official_candidate_or_google_button():
    card = job_card(job().model_dump(), {"candidate_job_url": "https://reolink.com/jobs/fake", "candidate_job_kind": "official_candidate"})
    card["company_careers_url"] = "https://reolink.com/careers"
    card["search_fallback_url"] = "https://www.google.com/search?q=Reolink"
    links = actionable_job_links(card)
    assert len(links) == 1
    assert links[0][0] == "Search LinkedIn ↗"
