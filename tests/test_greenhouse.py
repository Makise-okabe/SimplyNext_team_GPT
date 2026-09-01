from career_agent.tools import greenhouse


def test_greenhouse_board_slug_from_board_and_job_urls() -> None:
    assert greenhouse.greenhouse_board_slug("https://job-boards.greenhouse.io/reolink") == "reolink"
    assert greenhouse.greenhouse_board_slug("https://job-boards.greenhouse.io/reolink/jobs/123") == "reolink"
    assert greenhouse.greenhouse_board_slug("https://reolink.com/careers") is None


def test_fetch_greenhouse_jobs_parses_public_board_payload(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "jobs": [
                    {
                        "title": "Site Reliability Engineer (SRE)",
                        "absolute_url": "https://job-boards.greenhouse.io/reolink/jobs/123",
                        "location": {"name": "Singapore"},
                    }
                ]
            }

    monkeypatch.setattr(greenhouse.httpx, "get", lambda *args, **kwargs: FakeResponse())

    jobs = greenhouse.fetch_greenhouse_jobs("reolink")

    assert len(jobs) == 1
    assert jobs[0].title == "Site Reliability Engineer (SRE)"
    assert jobs[0].location == "Singapore"
    assert jobs[0].url.endswith("/jobs/123")
