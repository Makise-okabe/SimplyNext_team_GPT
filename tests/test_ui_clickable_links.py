from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_unverified_result_card_renders_clickable_official_and_linkedin_searches():
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
    assert any(label == "Search LinkedIn ↗" and "linkedin.com/jobs/search" in url for label, url in labels_and_urls)
    assert all("Nov%2FDec" not in url for _, url in labels_and_urls)
