from career_agent.job_research_quality import (
    clean_jd_text,
    is_plausible_official_url,
    page_is_closed,
)


def test_aggregator_is_not_primary_but_company_and_ats_are() -> None:
    assert not is_plausible_official_url(
        "https://sg.trabajo.org/job-3820-abc",
        "REC Solar Pte Ltd",
    )
    assert is_plausible_official_url(
        "https://careers.point72.com/CSJobDetail?jobCode=CPA-0014976",
        "Point72 Asia (Singapore Pte Ltd)",
    )
    assert is_plausible_official_url(
        "https://hire-r1.mokahr.com/social-recruitment/tesla/100004141#/job/abc",
        "Tesla",
    )


def test_closed_secondary_page_is_rejected_signal() -> None:
    assert page_is_closed("This role is No longer accepting applications")
    assert page_is_closed("Applications are closed for this position")
    assert not page_is_closed("Applications reviewed on a rolling basis")


def test_clean_jd_removes_linkedin_recommendation_tail() -> None:
    raw = """Site Reliability Engineer
Xiaomi Technology
Job Responsibilities
1. Keep global systems available.
Job Requirements
1. Python and Linux.
Similar jobs
Random Micron role
People also viewed
Random Google role
LinkedIn
© 2026
"""
    cleaned = clean_jd_text(raw)
    assert "Job Responsibilities" in cleaned
    assert "Python and Linux" in cleaned
    assert "Similar jobs" not in cleaned
    assert "Random Micron role" not in cleaned
    assert "Random Google role" not in cleaned
