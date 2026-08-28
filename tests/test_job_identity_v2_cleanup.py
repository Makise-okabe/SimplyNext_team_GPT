from career_agent.job_identity.discover_candidates import (
    _direct_candidates,
    build_progressive_queries,
)
from career_agent.models.job_identity import JobIdentity


def _mckinsey_identity() -> JobIdentity:
    return JobIdentity(
        source_message_id="m",
        signal_index=1,
        company="McKinsey & Company",
        title="Innovation and Learning Centre (ILC) Intern",
        location="Singapore",
        opportunity_type="internship",
        business_unit="Innovation and Learning Centre (ILC)",
        distinctive_phrases=[
            "Innovation and Learning Centre (ILC)",
            "Advanced Remanufacturing and Technology Center (ARTC)",
            "Supply Chain Control Tower",
        ],
        direct_urls=[
            "https://outlook.live.com/owa/?ItemID=abc",
            "https://aka.ms/example",
            "https://forms.office.com/r/example",
        ],
        identity_strength="strong",
        source_fingerprint="abc",
    )


def test_transport_links_are_removed_but_application_form_is_retained() -> None:
    candidates = _direct_candidates(_mckinsey_identity())

    assert "https://outlook.live.com/owa/?ItemID=abc" not in candidates
    assert "https://aka.ms/example" not in candidates
    assert "https://forms.office.com/r/example" in candidates


def test_queries_do_not_repeat_full_title_and_business_unit() -> None:
    queries = build_progressive_queries(_mckinsey_identity())

    assert queries[0][0] == "metadata"
    assert '"Innovation and Learning Centre"' in queries[0][1]
    assert "(ILC) Intern" not in queries[0][1]
    assert queries[1][0] == "distinctive_phrase"
    assert "Advanced Remanufacturing and Technology Center" in queries[1][1]
