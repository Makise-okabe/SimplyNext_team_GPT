from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_unverified_result_card_does_not_render_a_fake_search_link():
    card = {
        "company": "Goldilock",
        "title": "Embedded Software Engineer (Aug - Nov/Dec 2026)",
        "score": 95,
        "final_score": 95,
    }
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui" / "app.py")
    app.session_state["sn_result"] = {
        "metrics": {"active_jobs": 1},
        "live_inbox": {"candidate_count": 1},
        "top_matches": [card],
        "all_rankings": [card],
    }

    app.run(timeout=10)

    assert not app.exception
    labels_and_urls = {(button.label, button.url) for button in app.get("link_button")}
    assert not any("google.com/search" in url for _, url in labels_and_urls)
    assert not any("linkedin.com/jobs/search" in url for _, url in labels_and_urls)
    assert labels_and_urls == set()


def test_archived_exact_linkedin_result_renders_its_direct_role_url():
    url = "https://sg.linkedin.com/jobs/view/embedded-software-engineer-at-goldilock-secure-4378805035"
    card = {
        "company": "Goldilock",
        "title": "Embedded Software Engineer (Aug - Nov/Dec 2026)",
        "score": 87,
        "final_score": 87,
        "candidate_job_url": url,
        "candidate_job_kind": "secondary_archived",
        "candidate_job_reason": "Exact role page found, but the listing is closed",
        "link_verification_status": "closed",
        "link_checked_at": "2026-09-06T00:00:00+00:00",
        "jd_status": "fetched_secondary",
        "matching_evidence_level": "full_jd",
    }
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui" / "app.py")
    app.session_state["sn_result"] = {
        "metrics": {"active_jobs": 1, "full_jd": 1},
        "live_inbox": {"candidate_count": 1},
        "top_matches": [card],
        "all_rankings": [card],
    }

    app.run(timeout=10)

    assert not app.exception
    labels_and_urls = {(button.label, button.url) for button in app.get("link_button")}
    assert ("Open archived LinkedIn JD ↗", url) in labels_and_urls
    assert not any("/jobs/search" in target for _, target in labels_and_urls)
