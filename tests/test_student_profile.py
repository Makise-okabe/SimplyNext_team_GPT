from pathlib import Path

from career_agent import course_enrichment, student_profile
from career_agent.course_enrichment import CourseEnrichment, extract_module_codes, infer_course_skills
from career_agent.student_profile import build_student_profile


def test_extract_module_codes_ignores_academic_year_and_keeps_real_modules() -> None:
    text = """
    ACADEMIC YEAR 2025/2026 SEMESTER 1
    EE 2028 MICROCONTROLLER PROGRAMMING AND INTERFACING A 4.00
    CS2040DE DATA STRUCTURES AND ALGORITHMS A- 4.00
    EE2211 INTRODUCTION TO MACHINE LEARNING A 4.00
    """
    assert extract_module_codes(text) == ["EE2028", "CS2040DE", "EE2211"]


def test_course_title_rules_create_useful_skills() -> None:
    assert {"embedded systems", "c/c++", "digital design"} <= infer_course_skills(
        "Microcontroller Programming and Interfacing"
    )
    assert {"machine learning", "python", "data analysis"} <= infer_course_skills(
        "Introduction to Machine Learning"
    )
    assert "semiconductor" in infer_course_skills("Microelectronics Materials and Devices")


def test_build_student_profile_separates_explicit_and_course_skills(monkeypatch) -> None:
    def fake_enrich(codes):
        assert codes == ["EE2028", "EE2211"]
        return [
            CourseEnrichment(
                module_code="EE2028",
                title="Microcontroller Programming and Interfacing",
                description="embedded systems and microcontrollers",
                academic_year="2025-2026",
                source_url="https://example/EE2028",
                skills=("c/c++", "embedded systems", "digital design"),
                source_status="fetched_nusmods",
            ),
            CourseEnrichment(
                module_code="EE2211",
                title="Introduction to Machine Learning",
                description="machine learning",
                academic_year="2025-2026",
                source_url="https://example/EE2211",
                skills=("machine learning", "python", "data analysis"),
                source_status="fetched_nusmods",
            ),
        ]

    monkeypatch.setattr(student_profile, "enrich_courses", fake_enrich)
    profile = build_student_profile(
        resume_text="Python Cadence Virtuoso semiconductor design",
        transcript_text="EE2028 Microcontroller Programming and Interfacing\nEE2211 Introduction to Machine Learning",
    )

    assert "python" in profile.explicit_skills
    assert "eda/cadence" in profile.explicit_skills
    assert "semiconductor" in profile.explicit_skills
    assert "embedded systems" in profile.course_derived_skills
    assert "machine learning" in profile.course_derived_skills
    assert set(profile.explicit_skills) <= set(profile.all_skills)
    assert set(profile.course_derived_skills) <= set(profile.all_skills)


def test_enrich_courses_uses_cache_without_network(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_fetch(code):
        calls.append(code)
        return CourseEnrichment(
            module_code=code,
            title="Introduction to Internet of Things",
            description="IoT embedded systems",
            academic_year="2025-2026",
            source_url="https://example/CS3237",
            skills=("iot", "embedded systems"),
            source_status="fetched_nusmods",
        )

    monkeypatch.setattr(course_enrichment, "fetch_nusmods_course", fake_fetch)
    cache = tmp_path / "course_cache.json"

    first = course_enrichment.enrich_courses(["CS3237"], cache_path=cache)
    second = course_enrichment.enrich_courses(["CS3237"], cache_path=cache)

    assert calls == ["CS3237"]
    assert first[0].skills == ("iot", "embedded systems")
    assert second[0].skills == ("iot", "embedded systems")
    assert cache.exists()
