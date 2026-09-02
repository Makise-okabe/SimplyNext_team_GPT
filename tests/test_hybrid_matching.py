from career_agent.hybrid_matching import (
    build_job_skill_profile,
    build_student_skill_profile,
    extract_explicit_skills,
    infer_title_skills,
    score_match,
)


def test_short_skill_terms_do_not_match_inside_unrelated_words() -> None:
    assert "java" not in extract_explicit_skills("JavaScript React developer")
    assert "javascript/typescript" in extract_explicit_skills("JavaScript React developer")
    assert "rf/wireless" not in extract_explicit_skills("performance engineer")
    assert "rf/wireless" in extract_explicit_skills("RF Engineer")


def test_ai_engineer_title_infers_core_skills() -> None:
    skills = infer_title_skills("AI Engineer")
    assert {"python", "machine learning", "deep learning"}.issubset(skills)


def test_sre_title_infers_cloud_devops_and_software() -> None:
    skills = infer_title_skills("Site Reliability Engineer")
    assert {"devops", "cloud", "software engineering"}.issubset(skills)


def test_chip_design_application_engineer_infers_semiconductor_skills() -> None:
    skills = infer_title_skills("Chip Design / Application Engineer")
    assert {
        "semiconductor",
        "digital design",
        "analog circuits",
        "eda/cadence",
    }.issubset(skills)


def test_source_only_job_is_not_discarded() -> None:
    profile = build_job_skill_profile(
        {
            "company": "Nanyang Singtech",
            "title": "Chip Design / Application Engineer",
            "matching_evidence_level": "source_only",
            "source_evidence": "UG and PG applicants from EE and CE are welcome.",
            "jd_text": "",
        }
    )
    assert profile.evidence_level == "source_only"
    assert "semiconductor" in profile.skills
    assert "eda/cadence" in profile.skills
    assert profile.inferred_skills


def test_full_jd_uses_jd_as_primary_evidence() -> None:
    jd = (
        "Responsibilities include Python data analysis and machine learning. "
        "Requirements include SQL and strong communication skills."
    )
    profile = build_job_skill_profile(
        {
            "company": "Example",
            "title": "Graduate Analyst",
            "matching_evidence_level": "full_jd",
            "jd_text": jd,
            "source_evidence": "short email text",
        }
    )
    assert profile.evidence_text == jd
    assert {"python", "data analysis", "machine learning", "sql", "communication"}.issubset(
        set(profile.skills)
    )


def test_course_skill_text_enriches_student_profile() -> None:
    student = build_student_skill_profile(
        resume_text="Embedded systems project using C++.",
        transcript_text="EE2026",
        course_skill_texts=[
            "Digital logic design using Verilog HDL and FPGA implementation."
        ],
    )
    assert "embedded systems" in student.skills
    assert "c/c++" in student.skills
    assert "digital design" in student.skills
    assert "verilog/hdl" in student.skills
    assert "fpga" in student.skills


def test_match_score_uses_job_skill_coverage() -> None:
    student = build_student_skill_profile(
        extra_skills=["python", "machine learning", "deep learning"]
    )
    job = build_job_skill_profile(
        {
            "company": "Reolink",
            "title": "AI Engineer",
            "matching_evidence_level": "source_only",
            "source_evidence": "Reolink available role: AI Engineer",
        }
    )
    result = score_match(student, job)
    assert result.score == 100.0
    assert set(result.matched_skills) == set(result.job_skills)
    assert result.missing_skills == ()
    assert result.confidence == "medium"


def test_empty_job_skills_score_zero_without_crashing() -> None:
    student = build_student_skill_profile(extra_skills=["python"])
    job = build_job_skill_profile(
        {
            "company": "Example",
            "title": "General Opportunity",
            "matching_evidence_level": "source_only",
            "source_evidence": "General opportunity",
        }
    )
    result = score_match(student, job)
    assert result.score == 0.0
    assert result.job_skills == ()
    assert result.confidence == "low"
