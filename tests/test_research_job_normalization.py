from career_agent.nodes.research_job import ResearchedJob, normalize_researched_job


def test_null_optional_fields_do_not_fail_normalization() -> None:
    item = ResearchedJob(
        company="McKinsey & Company",
        title="Innovation and Learning Centre (ILC) Intern",
        opportunity_type=None,
        raw_description=None,
        degree_requirements=None,
        required_skills=None,
        preferred_skills=None,
        evidence=None,
    )

    payload = normalize_researched_job(
        item,
        {"opportunity_type": "internship", "location": "Singapore"},
    )

    assert payload["opportunity_type"] == "internship"
    assert payload["raw_description"] == ""
    assert payload["degree_requirements"] == []
    assert payload["required_skills"] == []
    assert payload["preferred_skills"] == []
    assert payload["evidence"] == []


def test_microsoft_form_is_application_url_not_official_url() -> None:
    form_url = "https://forms.office.com/r/kWP6hTFRTz"
    item = ResearchedJob(
        company="McKinsey & Company",
        title="Innovation and Learning Centre (ILC) Intern",
        opportunity_type="internship",
        official_url=form_url,
    )

    payload = normalize_researched_job(item, {})

    assert payload["official_url"] is None
    assert payload["application_url"] == form_url
