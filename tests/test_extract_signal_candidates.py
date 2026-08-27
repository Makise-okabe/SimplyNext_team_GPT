from career_agent.nodes.extract_signal import build_candidate_chunks, is_candidate_url


def test_candidate_url_filter_removes_noise() -> None:
    assert not is_candidate_url("https://outlook.office365.com/owa/?ItemID=abc")
    assert not is_candidate_url("https://reufe.stripocdn.email/content/images/logo.png")
    assert not is_candidate_url("https://www.instagram.com/nuscfg/")
    assert not is_candidate_url("https://nus.edu.sg/cfg/events")


def test_candidate_url_filter_keeps_actionable_links() -> None:
    assert is_candidate_url("https://nomuracampus.tal.net/job/1006")
    assert is_candidate_url("https://nus-csm.symplicity.com/students/app/quicksearch?query=230167")
    assert is_candidate_url("https://go.gov.sg/dsta-brainhack-signup")


def test_build_candidate_chunks_uses_nearby_context_and_deduplicates() -> None:
    url = "https://example.com/jobs/123"
    text = f"AMD Product Development Engineer Singapore. Apply here: {url} by Friday."

    candidates = build_candidate_chunks(
        text,
        [url, url, "https://outlook.office365.com/owa/?ItemID=abc"],
    )

    assert len(candidates) == 1
    assert candidates[0].url == url
    assert "AMD Product Development Engineer" in candidates[0].context
    assert "Friday" in candidates[0].context
