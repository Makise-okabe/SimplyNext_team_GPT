from career_agent.job_identity.liveness import (
    _looks_like_generic_redirect,
    classify_liveness,
)


def test_404_is_closed() -> None:
    assert classify_liveness(404, "")[0] == "closed"


def test_expired_marker_is_closed_even_on_200() -> None:
    status, _ = classify_liveness(200, "This job is no longer available")
    assert status == "closed"


def test_apply_now_marker_is_open() -> None:
    status, _ = classify_liveness(200, "Ready? Apply now for this position")
    assert status == "open"


def test_plain_200_is_unknown_not_assumed_open() -> None:
    status, _ = classify_liveness(200, "Welcome to our careers site")
    assert status == "unknown"


def test_deep_job_url_redirect_to_careers_root_is_generic() -> None:
    assert _looks_like_generic_redirect(
        "https://example.com/careers/jobs/12345",
        "https://example.com/careers",
    ) is True
