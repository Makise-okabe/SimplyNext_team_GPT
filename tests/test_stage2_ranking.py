from career_agent import stage2_ranking
from career_agent.stage2_ranking import (
    SemanticAssessment,
    SemanticAssessmentBatch,
    build_stage2_prompt,
    rerank_stage1,
)


class FakeLLM:
    def __init__(self, batches):
        if isinstance(batches, list):
            self.batches = list(batches)
        else:
            self.batches = [batches]
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.batches:
            value = self.batches.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        return SemanticAssessmentBatch(assessments=[])


def _student():
    return {
        "explicit_skills": ["python", "semiconductor", "eda/cadence", "c/c++"],
        "course_derived_skills": ["machine learning", "digital design", "embedded systems"],
        "course_enrichment": [
            {
                "module_code": "EE2028",
                "title": "Microcontroller Programming and Interfacing",
                "skills": ["embedded systems", "c/c++", "digital design"],
            }
        ],
    }


def _stage1():
    return [
        {
            "company": "Chip Co",
            "title": "Chip Design Engineer",
            "score": 91.0,
            "confidence": "medium",
            "evidence_level": "source_only",
            "matched_resume_skills": ["semiconductor", "eda/cadence"],
            "matched_course_skills": ["digital design"],
            "inferred_job_skills": ["semiconductor", "digital design"],
        },
        {
            "company": "Marketing Co",
            "title": "Brand Marketing Intern",
            "score": 84.0,
            "confidence": "low",
            "evidence_level": "source_only",
            "matched_resume_skills": [],
            "matched_course_skills": [],
            "inferred_job_skills": ["marketing", "communication"],
        },
    ]


def _jobs():
    return [
        {
            "company": "Chip Co",
            "title": "Chip Design Engineer",
            "matching_evidence_level": "source_only",
            "source_evidence": "Chip design role for engineering students",
            "official_job_url": "https://chip.example/job",
            "application_url": "https://chip.example/apply",
        },
        {
            "company": "Marketing Co",
            "title": "Brand Marketing Intern",
            "matching_evidence_level": "source_only",
            "source_evidence": "Brand marketing internship",
            "official_job_url": "https://marketing.example/job",
        },
    ]


def _assessment(index: int, score: float = 80) -> SemanticAssessment:
    return SemanticAssessment(
        candidate_index=index,
        semantic_score=score,
        fit_label="good",
        why_match="Grounded fit.",
        matched_evidence=[],
        missing_or_weak_evidence=[],
        confidence="medium",
    )


def test_stage2_receives_email_context_and_verified_web_jd_together():
    ranked = [_stage1()[0]]
    source = [_jobs()[0] | {
        "matching_evidence_level": "full_jd",
        "source_evidence": "NUS email: Chip Design Engineer, TC ID 1234",
        "jd_text": "Official JD: design analog and digital semiconductor circuits.",
    }]
    prompt = build_stage2_prompt(
        resume_text="Cadence and semiconductor experience",
        student_profile=_student(),
        stage1_top=ranked,
        source_jobs=source,
    )
    assert "NUS email: Chip Design Engineer, TC ID 1234" in prompt
    assert "Official JD: design analog and digital semiconductor circuits." in prompt
    assert '"email_evidence"' in prompt
    assert '"verified_web_job_description"' in prompt


def test_stage2_batches_candidates_and_semantic_score_drives_order():
    llm = FakeLLM(
        [
            SemanticAssessmentBatch(
                assessments=[
                    SemanticAssessment(
                        candidate_index=0,
                        semantic_score=94,
                        fit_label="strong",
                        why_match="Direct semiconductor and Cadence evidence.",
                        matched_evidence=["Cadence Virtuoso", "semiconductor experience"],
                        missing_or_weak_evidence=["Job evidence is source-only"],
                        confidence="medium",
                    ),
                    SemanticAssessment(
                        candidate_index=1,
                        semantic_score=28,
                        fit_label="weak",
                        why_match="Mostly generic overlap and role-family mismatch.",
                        matched_evidence=[],
                        missing_or_weak_evidence=["No marketing experience"],
                        confidence="medium",
                    ),
                ]
            )
        ]
    )

    results = rerank_stage1(
        resume_text="Cadence semiconductor C++ embedded research",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        stage1_top_n=20,
        batch_size=5,
        llm=llm,
    )

    assert len(llm.calls) == 1
    assert all(item.semantic_assessed for item in results)
    assert results[0].company == "Chip Co"
    assert results[0].final_score > results[1].final_score
    assert results[0].application_url == "https://chip.example/apply"
    assert results[0].official_job_url == "https://chip.example/job"


def test_stage2_retries_missing_assessment_and_restores_full_coverage():
    llm = FakeLLM(
        [
            SemanticAssessmentBatch(assessments=[_assessment(0, 90)]),
            SemanticAssessmentBatch(assessments=[_assessment(0, 30)]),
        ]
    )

    results = rerank_stage1(
        resume_text="Cadence semiconductor",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        batch_size=5,
        llm=llm,
    )

    assert len(llm.calls) == 2
    assert len(results) == 2
    assert all(item.semantic_assessed for item in results)
    marketing = next(item for item in results if item.company == "Marketing Co")
    assert marketing.semantic_score == 30.0


def test_stage2_recovers_whole_failed_batch_with_individual_retries():
    stage1 = _stage1() * 3
    jobs = _jobs() * 3
    llm = FakeLLM(
        [
            SemanticAssessmentBatch(assessments=[]),
            *[
                SemanticAssessmentBatch(assessments=[_assessment(0, 80 - index)])
                for index in range(5)
            ],
            SemanticAssessmentBatch(assessments=[_assessment(0, 70)]),
        ]
    )

    results = rerank_stage1(
        resume_text="technical resume",
        student_profile=_student(),
        all_jobs=jobs,
        stage1_rankings=stage1,
        stage1_top_n=6,
        batch_size=5,
        llm=llm,
    )

    assert len(llm.calls) == 7
    assert len(results) == 6
    assert all(item.semantic_assessed for item in results)


def test_stage2_rate_limit_retries_with_backoff_without_real_sleep():
    class RateLimitError(Exception):
        pass

    llm = FakeLLM(
        [
            RateLimitError("rate limit exceeded"),
            SemanticAssessmentBatch(assessments=[_assessment(0, 90), _assessment(1, 70)]),
        ]
    )
    sleeps = []

    results = rerank_stage1(
        resume_text="technical resume",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        batch_size=5,
        llm=llm,
        sleep_fn=sleeps.append,
    )

    assert len(llm.calls) == 2
    assert sleeps == [stage2_ranking.STAGE2_RATE_LIMIT_FALLBACK_SECONDS]
    assert all(item.semantic_assessed for item in results)


def test_stage2_caps_fallback_after_retry_failure():
    llm = FakeLLM(
        [
            SemanticAssessmentBatch(assessments=[]),
            SemanticAssessmentBatch(assessments=[]),
            SemanticAssessmentBatch(assessments=[]),
        ]
    )

    results = rerank_stage1(
        resume_text="technical resume",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        batch_size=5,
        llm=llm,
    )

    assert len(llm.calls) == 3
    assert all(not item.semantic_assessed for item in results)
    assert all(item.final_score <= 60.0 for item in results)
    assert all(not item.missing_or_weak_evidence for item in results)  # A service outage is not a student skill gap.


def test_stage2_small_batches_use_multiple_calls_without_dropping_candidates():
    stage1 = _stage1() * 3
    jobs = _jobs() * 3

    batches = []
    for size in (2, 2, 2):
        batches.append(
            SemanticAssessmentBatch(
                assessments=[_assessment(index, 80 - index) for index in range(size)]
            )
        )
    llm = FakeLLM(batches)

    results = rerank_stage1(
        resume_text="technical resume",
        student_profile=_student(),
        all_jobs=jobs,
        stage1_rankings=stage1,
        stage1_top_n=6,
        batch_size=2,
        llm=llm,
    )

    assert len(llm.calls) == 3
    assert len(results) == 6
    assert all(item.semantic_assessed for item in results)


def test_stage2_build_llm_uses_json_schema_not_function_calling(monkeypatch):
    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["structured"] = kwargs
            return "json-schema-model"

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(stage2_ranking, "ChatGroq", FakeChatGroq)

    model = stage2_ranking._build_llm()

    assert model == "json-schema-model"
    assert captured["schema"] is SemanticAssessmentBatch
    assert captured["structured"]["method"] == "json_schema"
    assert captured["structured"]["strict"] is False
