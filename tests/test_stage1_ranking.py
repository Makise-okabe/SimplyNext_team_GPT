from career_agent.stage1_ranking import rank_job, rank_jobs


def _student():
    return {
        "explicit_skills": ["python", "semiconductor", "eda/cadence"],
        "course_derived_skills": ["machine learning", "digital design", "analog circuits"],
    }


def test_resume_explicit_skills_are_weighted_above_course_only_skills():
    explicit_job = {
        "company": "A",
        "title": "Python Engineer",
        "source_evidence": "Python software engineer",
        "matching_evidence_level": "source_only",
    }
    course_job = {
        "company": "B",
        "title": "Digital Design Engineer",
        "source_evidence": "Digital design engineer",
        "matching_evidence_level": "source_only",
    }

    explicit_score = rank_job(_student(), explicit_job).score
    course_score = rank_job(_student(), course_job).score

    assert explicit_score > course_score


def test_full_jd_changes_confidence_not_skill_overlap_score():
    source_job = {
        "company": "A",
        "title": "AI Engineer",
        "source_evidence": "Python machine learning",
        "matching_evidence_level": "source_only",
    }
    full_job = {
        **source_job,
        "matching_evidence_level": "full_jd",
        "jd_text": "Python machine learning",
    }

    source = rank_job(_student(), source_job)
    full = rank_job(_student(), full_job)

    assert full.score == source.score
    assert full.confidence == "high"


def test_inferred_job_skills_are_preserved_for_explanation():
    job = {
        "company": "Nanyang Singtech",
        "title": "Chip Design / Application Engineer",
        "source_evidence": "",
        "matching_evidence_level": "source_only",
    }

    result = rank_job(_student(), job)

    assert "semiconductor" in result.job_skills
    assert "eda/cadence" in result.job_skills
    assert result.inferred_job_skills


def test_rank_jobs_sorts_best_match_first():
    jobs = [
        {
            "company": "Marketing Co",
            "title": "Brand Marketing Intern",
            "source_evidence": "brand marketing",
            "matching_evidence_level": "source_only",
        },
        {
            "company": "Chip Co",
            "title": "Chip Design Engineer",
            "source_evidence": "semiconductor Cadence digital design",
            "matching_evidence_level": "source_only",
        },
    ]

    ranked = rank_jobs(_student(), jobs)

    assert ranked[0].company == "Chip Co"
    assert ranked[0].score > ranked[1].score
