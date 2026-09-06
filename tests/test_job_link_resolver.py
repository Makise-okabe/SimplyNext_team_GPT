import httpx
import pytest
from career_agent.tools.web_fetch import FetchedPage

@pytest.fixture(autouse=True)
def fetched_search_destinations(monkeypatch):
    # Search metadata alone is no longer enough: these positive cases supply a page.
    def fetch(url, **kwargs):
        title = "AI Engineering role - Reolink" if "654321" in url else "AI Engineer - Reolink"
        return FetchedPage(url, url, 200, title, title + "\nResponsibilities\n" + "Build machine learning systems with Python. " * 20 + "\nRequirements\nEngineering degree.")
    monkeypatch.setattr(job_link_resolver, "fetch_public_page", fetch)

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
    assert len(calls) == 3  # A secondary hit must not stop official discovery.
    assert calls[0][0] == '"Reolink" "AI Engineer"'


def test_resolver_retains_matching_candidate_when_page_blocks_automation(monkeypatch):
    url = "https://reolink.com/careers/jobs/ai-engineer"
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [SearchResult(
            title="AI Engineer - Reolink Careers",
            url=url,
            snippet="Official Reolink careers page",
        )],
    )
    monkeypatch.setattr(
        job_link_resolver,
        "fetch_public_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("access challenge")),
    )

    resolved, result = resolve_job_link(_job())

    assert result.url is None
    assert resolved.candidate_job_url == url
    assert resolved.candidate_job_kind == "official_candidate"
    assert resolved.search_fallback_url is None
    assert resolved.search_resolution_status == "unresolved"


def test_resolver_navigates_homepage_to_careers_to_exact_role(monkeypatch):
    home = "https://www.bhglobal.com.sg/"
    careers = "https://www.bhglobal.com.sg/join-us/"
    exact = "https://www.bhglobal.com.sg/jobs/electrical-intern/"
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [SearchResult("Home : BH Global Corporation Ltd", home, "BH Global")],
    )

    def fetch(url, **kwargs):
        if url == home:
            return FetchedPage(url, url, 200, "BH Global", "BH Global", (careers,), (), (), "html", ((careers, "Join Us"),))
        if url == careers:
            return FetchedPage(url, url, 200, "Join Us", "BH Global careers", (exact,), (), (), "html", ((exact, "Electrical Intern"),))
        assert url == exact
        text = "Electrical Intern\nJob Scope\n" + "Assist with electrical systems and testing. " * 20 + "\nJob requirement\nElectrical engineering student."
        return FetchedPage(url, url, 200, "Electrical Intern", text, (), ("Electrical Intern",))

    monkeypatch.setattr(job_link_resolver, "fetch_public_page", fetch)
    resolved, result = resolve_job_link(_job("BH Global Corporation Ltd", "Electrical Intern"))
    assert result.url == exact
    assert result.kind == "official_exact"
    assert resolved.jd_status == "fetched_official"
    assert [attempt["status"] for attempt in resolved.link_attempts] == ["generic_page", "generic_page", "verified"]


def test_resolver_search_keeps_broad_official_homepage(monkeypatch):
    homepage = SearchResult(
        "Home : BH Global Corporation Ltd",
        "https://www.bhglobal.com.sg/",
        "BH Global",
    )
    captured = {}

    def aggregate(query, **kwargs):
        captured.update(kwargs)
        return [homepage]

    monkeypatch.setattr(job_link_resolver, "search_public_web_aggregated", aggregate)
    assert job_link_resolver.search_public_web('"BH Global Corporation Ltd" "Electrical Intern"') == [homepage]
    assert captured["strict_relevance"] is False
    assert captured["max_results"] >= 12


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


@pytest.mark.parametrize("url", [
    "https://www.mycareersfuture.gov.sg/job/engineering/full-stack-java-developer-conex-healthcare-071e6038cf90444da907ff35beee6cae",
    "https://www.foundit.sg/job/full-stack-java-developer-conex-healthcare-pte-ltd-singapore-11982036",
    "https://glints.com/opportunities/s/full-stack-java-developer/f3ee58ae-d300-428b-a7cb-4d283ffcca7b",
])
def test_resolver_accepts_verified_specific_singapore_job_platforms(monkeypatch, url):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [SearchResult(
            "Full Stack Java Developer - CoNEX Healthcare Pte Ltd",
            url,
            "Singapore full-time job",
        )],
    )
    text = "Full Stack Java Developer\nCoNEX Healthcare Pte Ltd\nResponsibilities\n" + "Build Java web applications and APIs. " * 20 + "\nRequirements\nJava development experience."
    monkeypatch.setattr(
        job_link_resolver,
        "fetch_public_page",
        lambda target, **kwargs: FetchedPage(target, target, 200, "Full Stack Java Developer - CoNEX Healthcare", text, (), ("Full Stack Java Developer",)),
    )
    resolved, result = resolve_job_link(_job("CoNEX Healthcare Pte Ltd", "Full Stack Java Developer"))
    assert result.url == url
    assert result.kind == "secondary_exact"
    assert resolved.job_page_confidence == "medium"


def test_resolver_uses_one_grounded_search_after_normal_search_misses(monkeypatch):
    url = "https://www.foundit.sg/job/full-stack-java-developer-conex-healthcare-pte-ltd-singapore-11982036"
    calls = []
    monkeypatch.setattr(job_link_resolver, "search_public_web", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        job_link_resolver,
        "search_groq_grounded",
        lambda query, max_results=10: calls.append(query) or [SearchResult(
            "Full Stack Java Developer - CoNEX Healthcare Pte Ltd",
            url,
            "Singapore full-time role",
        )],
    )
    text = "Full Stack Java Developer\nCoNEX Healthcare Pte Ltd\nResponsibilities\n" + "Build Java web applications and APIs. " * 20 + "\nRequirements\nJava development experience."
    monkeypatch.setattr(
        job_link_resolver,
        "fetch_public_page",
        lambda target, **kwargs: FetchedPage(target, target, 200, "Full Stack Java Developer - CoNEX Healthcare", text, (), ("Full Stack Java Developer",)),
    )
    resolved, result = resolve_job_link(_job("CoNEX Healthcare Pte Ltd", "Full Stack Java Developer"))
    assert len(calls) == 1
    assert result.url == url
    assert resolved.link_verification_status == "verified"


def test_resolver_follows_lenovo_dynamic_board_to_official_job(monkeypatch):
    board = "https://talent.lenovo.com.cn/position?keyword=AI+Solution+Architect"
    detail = "https://talent.lenovo.com.cn/position/detail?id=2399"
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [SearchResult(
            "Lenovo recruitment positions",
            board,
            "AI Solution Architect Global Future Leaders Beijing",
        )],
    )

    def fetch(url, **kwargs):
        if url == board:
            return FetchedPage(
                url,
                url,
                200,
                "Lenovo Campus Recruitment",
                "Recruitment positions",
                (detail,),
                (),
                (),
                "public_career_api",
                ((detail, "AI Solution Architect Global Future Leaders SSG"),),
            )
        description = (
            "Responsibilities\nDevelop and deploy RAG and LLM applications with Python. " * 8
            + "\nRequirements\nPyTorch or TensorFlow, Linux and Git."
        )
        return FetchedPage(
            url,
            url,
            200,
            "Lenovo Campus Recruitment",
            "",
            (),
            (),
            ({
                "@type": "JobPosting",
                "title": "AI Solution Architect",
                "description": description,
                "hiringOrganization": {"name": "Lenovo"},
                "identifier": "2399",
            },),
            "public_ats_detail",
        )

    monkeypatch.setattr(job_link_resolver, "fetch_public_page", fetch)
    resolved, result = resolve_job_link(
        _job("Lenovo China", "Global Future Leaders: AI Solution Architect")
    )

    assert result.url == detail
    assert result.kind == "official_exact"
    assert resolved.job_page_confidence == "high"
    assert resolved.job_id == "2399"


def test_resolver_opens_supported_lenovo_board_when_search_never_returns_it(monkeypatch):
    board = "https://talent.lenovo.com.cn/position?keyword=AI+Solution+Architect"
    detail = "https://talent.lenovo.com.cn/position/detail?id=9876"
    search_calls = []
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: search_calls.append(query) or [SearchResult(
            "Lenovo support", "https://support.lenovo.com/sg/en/", "Support"
        )],
    )

    def fetch(url, **kwargs):
        if url == board:
            return FetchedPage(
                url, url, 200, "Lenovo Campus Recruitment", "Recruitment positions",
                (detail,), (), (), "public_career_api",
                ((detail, "AI Solution Architect Global Future Leaders SSG"),),
            )
        if url == detail:
            description = (
                "Responsibilities\nDevelop RAG and LLM applications using Python. " * 8
                + "\nRequirements\nPyTorch or TensorFlow, Linux and Git."
            )
            return FetchedPage(
                url, url, 200, "Lenovo Campus Recruitment", "", (), (), ({
                    "@type": "JobPosting",
                    "title": "AI Solution Architect",
                    "description": description,
                    "hiringOrganization": {"name": "Lenovo"},
                    "identifier": "9876",
                },), "public_ats_detail",
            )
        raise AssertionError(f"unexpected fetch: {url}")

    monkeypatch.setattr(job_link_resolver, "fetch_public_page", fetch)
    resolved, result = resolve_job_link(
        _job("Lenovo China", "Global Future Leaders: AI Solution Architect")
    )

    assert result.url == detail
    assert result.kind == "official_exact"
    assert resolved.job_id == "9876"
    assert search_calls == []
    assert resolved.link_attempts[0]["query"] == "supported_official_career_board"


def test_resolver_reserves_fetches_for_grounded_dynamic_board(monkeypatch):
    board = "https://talent.lenovo.com.cn/position?projectType=3"
    detail = "https://talent.lenovo.com.cn/position/detail?id=2399"
    weak = [
        SearchResult("Lenovo support", "https://support.lenovo.com/sg/en/", "Support"),
        SearchResult("Lenovo PCs", "https://www.lenovo.com/sg/en/pc/", "Products"),
        SearchResult("Solution Architect", "https://jobs.lenovo.com/en_US/careers/JobDetail/Solution-Architect/76610", "Different role"),
        SearchResult("AI Technical Architect", "https://jobs.lenovo.com/en_US/careers/JobDetail/AI-Technical-Architect/74165", "Different role"),
    ]
    monkeypatch.setattr(job_link_resolver, "search_public_web", lambda *args, **kwargs: weak)
    monkeypatch.setattr(
        job_link_resolver,
        "search_groq_grounded",
        lambda *args, **kwargs: [SearchResult(
            "AI Solution Architect - Global Future Leaders",
            board,
            "Lenovo China Beijing",
        )],
    )
    fetched = []

    def fetch(url, **kwargs):
        fetched.append(url)
        if url == board:
            return FetchedPage(
                url, url, 200, "Lenovo Campus Recruitment", "Recruitment positions",
                (detail,), (), (), "public_career_api",
                ((detail, "AI Solution Architect Global Future Leaders SSG"),),
            )
        if url == detail:
            description = (
                "Responsibilities\nDevelop RAG and LLM applications using Python. " * 8
                + "\nRequirements\nPyTorch or TensorFlow, Linux and Git."
            )
            return FetchedPage(
                url, url, 200, "Lenovo Campus Recruitment", "", (), (), ({
                    "@type": "JobPosting",
                    "title": "AI Solution Architect",
                    "description": description,
                    "hiringOrganization": {"name": "Lenovo"},
                    "identifier": "2399",
                },), "public_ats_detail",
            )
        raise httpx.HTTPStatusError(
            "blocked",
            request=httpx.Request("GET", url),
            response=httpx.Response(403),
        )

    monkeypatch.setattr(job_link_resolver, "fetch_public_page", fetch)
    resolved, result = resolve_job_link(
        _job("Lenovo China", "Global Future Leaders: AI Solution Architect")
    )

    assert result.url == detail
    assert result.kind == "official_exact"
    assert board in fetched and detail in fetched
    assert "https://support.lenovo.com/sg/en/" not in fetched
    assert "https://www.lenovo.com/sg/en/pc/" not in fetched
    assert len(fetched) <= job_link_resolver.TOTAL_FETCH_LIMIT


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
    assert resolved.search_resolution_status == "unresolved"


def test_closed_exact_linkedin_is_kept_as_archived_evidence(monkeypatch):
    url = "https://sg.linkedin.com/jobs/view/embedded-software-engineer-at-goldilock-secure-4378805035"
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web",
        lambda query, **kwargs: [
            SearchResult(
                title="Embedded Software Engineer - Goldilock Secure",
                url=url,
                snippet="No longer accepting applications",
            )
        ],
    )
    monkeypatch.setattr(
        job_link_resolver,
        "fetch_public_page",
        lambda url, **kwargs: FetchedPage(
            url,
            url,
            200,
            "Embedded Software Engineer - Goldilock Secure",
            "Embedded Software Engineer\nGoldilock Secure\n"
            "No longer accepting applications\nResponsibilities\n"
            + "Develop embedded C and C++ systems. " * 20
            + "\nRequirements\nEmbedded engineering experience.",
        ),
    )

    resolved, result = resolve_job_link(
        _job("Goldilock", "Embedded Software Engineer (Aug - Nov/Dec 2026)")
    )

    assert result.url is None
    assert resolved.job_page_url is None
    assert resolved.candidate_job_url == url
    assert resolved.candidate_job_kind == "secondary_archived"
    assert resolved.jd_status == "fetched_secondary"
    assert "embedded C and C++" in resolved.jd_text
