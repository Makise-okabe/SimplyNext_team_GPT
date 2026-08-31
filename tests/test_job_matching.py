from career_agent import job_matching
from career_agent.models.job_record import JobRecord
from career_agent.models.match_result import MatchResult
from career_agent.models.student_profile import StudentProfile


class _FakeStructuredLLM:
    def __init__(self, result: MatchResult):
        self.result = result

    def invoke(self, _prompt: str) -> MatchResult:
        return self.result


def _job(company: str, title: str) -> JobRecord:
    return JobRecord(
        source_message_id="m1",
        source_subject="Career email",
        company=company,
        title=title,
        jd_status="fetched_secondary",
        jd_source_url="https://www.linkedin.com/jobs/view/123",
        jd_text="Python embedded systems engineer requirements " * 30,
    )


def test_match_job_normalizes_recommendation_and_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        job_matching,
        "_build_llm",
        lambda: _FakeStructuredLLM(
            MatchResult(
                score=91,
                recommendation="possible_match",
                matched_strengths=["Python"],
                rationale="Strong evidence.",
            )
        ),
    )
    job = _job("Example", "Engineer").model_copy(
        update={"primary_source_url": "https://example.com/jobs/1"}
    )
    result = job_matching.match_job(StudentProfile(skills=["Python"]), job)
    assert result.score == 91
    assert result.recommendation == "strong_match"
    assert result.company == "Example"
    assert result.primary_source_url == "https://example.com/jobs/1"


def test_rank_jobs_sorts_highest_score_first(monkeypatch) -> None:
    scores = iter([62, 88])

    def fake_match(_profile, job):
        score = next(scores)
        return MatchResult(
            company=job.company,
            title=job.title,
            score=score,
            recommendation=job_matching._recommendation_for_score(score),
        )

    monkeypatch.setattr(job_matching, "match_job", fake_match)
    ranked = job_matching.rank_jobs(
        StudentProfile(),
        [_job("Low", "Role A"), _job("High", "Role B")],
    )
    assert [item.company for item in ranked] == ["High", "Low"]
