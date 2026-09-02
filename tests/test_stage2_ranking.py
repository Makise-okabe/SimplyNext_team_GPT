from career_agent.stage2_ranking import (
    SemanticAssessment,
    SemanticAssessmentBatch,
    rerank_stage1,
)


class FakeLLM:
    def __init__(self, batch):
        self.batch = batch
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return self.batch


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


def test_stage2_uses_one_llm_call_and_semantic_score_drives_order():
    llm = FakeLLM(
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
    )

    results = rerank_stage1(
        resume_text="Cadence semiconductor C++ embedded research",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        stage1_top_n=20,
        llm=llm,
    )

    assert len(llm.calls) == 1
    assert results[0].company == "Chip Co"
    assert results[0].final_score > results[1].final_score
    assert results[0].application_url == "https://chip.example/apply"
    assert results[0].official_job_url == "https://chip.example/job"


def test_stage2_preserves_all_candidates_even_when_llm_omits_one():
    llm = FakeLLM(
        SemanticAssessmentBatch(
            assessments=[
                SemanticAssessment(
                    candidate_index=0,
                    semantic_score=90,
                    fit_label="strong",
                    why_match="Strong technical fit.",
                    matched_evidence=["Cadence"],
                    missing_or_weak_evidence=[],
                    confidence="medium",
                )
            ]
        )
    )

    results = rerank_stage1(
        resume_text="Cadence semiconductor",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        llm=llm,
    )

    assert len(results) == 2
    marketing = next(item for item in results if item.company == "Marketing Co")
    assert marketing.semantic_score == 84.0
    assert marketing.confidence == "low"
    assert "LLM semantic assessment missing" in marketing.missing_or_weak_evidence


def test_stage2_ignores_duplicate_and_out_of_range_assessments():
    llm = FakeLLM(
        SemanticAssessmentBatch(
            assessments=[
                SemanticAssessment(
                    candidate_index=0,
                    semantic_score=88,
                    fit_label="strong",
                    why_match="First assessment wins.",
                    matched_evidence=[],
                    missing_or_weak_evidence=[],
                    confidence="medium",
                ),
                SemanticAssessment(
                    candidate_index=0,
                    semantic_score=5,
                    fit_label="weak",
                    why_match="Duplicate must be ignored.",
                    matched_evidence=[],
                    missing_or_weak_evidence=[],
                    confidence="low",
                ),
                SemanticAssessment(
                    candidate_index=99,
                    semantic_score=100,
                    fit_label="strong",
                    why_match="Out of range.",
                    matched_evidence=[],
                    missing_or_weak_evidence=[],
                    confidence="high",
                ),
            ]
        )
    )

    results = rerank_stage1(
        resume_text="technical resume",
        student_profile=_student(),
        all_jobs=_jobs(),
        stage1_rankings=_stage1(),
        llm=llm,
    )

    chip = next(item for item in results if item.company == "Chip Co")
    assert chip.semantic_score == 88.0
