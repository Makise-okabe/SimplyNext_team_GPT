from datetime import datetime, timezone

from career_agent.all_job_extraction import ExtractionMetrics
from career_agent.goh_extraction import _parse_deadline, extract_goh_opportunities


def _empty_base(**kwargs):
    return [], ExtractionMetrics(llm_calls=0, source_chars=len(kwargs["corpus"])), []


def test_current_goh_multi_role_rows_and_direct_urls() -> None:
    point72 = "https://careers.point72.com/CSJobDetail?jobName=point72-academy-investment-analyst-program-for-upcoming-graduates-2027-sg-&jobCode=CPA-0014976"
    corpus = f"""SOURCE: EMAIL
JOBS
[[SIMPLYNEXT_TABLE_START]]
INDUSTRY | COMPANY | ROLE | TC ID | REMARKS
Engineering and Manufacturing | Ley Choon Group Holdings | 1. Management Associate 2. Engineer Associate 3. EHS Officer 4. Assistant Engineer 5. Project Engineer | 1. 6a7d6ca02c3cca001d178abf 2. 6a7a98d92c3cca001dfe99ef 3. 6a7ab5cc9787d8001d0faa77 4. 6a7ab66b2c3cca001dff6867 5. 6a7196d8c5eac4001d32db4b | Deadline: 12 Nov 2026 UG – CDE CE ME EE
Fund Management | Point72 Asia (Singapore Pte Ltd) | Point72 Academy Investment Analyst Program for Upcoming Graduates (2027 – Singapore) | Apply via link <{point72}> | Applications reviewed on a rolling basis. UG & PG (Master’s) – All
Information Communication Technology | AvePoint Singapore | Backend Developer | 6a7d298d2c3cca001d1462c2 | Deadline: 31 Aug Full-Time Job UG, PG (Master) - CDE Eng, SoC
[[SIMPLYNEXT_TABLE_END]]
"""
    opportunities, metrics, errors = extract_goh_opportunities(
        source_name="Goh Ze Li",
        source_message_id="latest-goh",
        source_date=datetime(2026, 8, 27, 23, 30, tzinfo=timezone.utc),
        corpus=corpus,
        base_extractor=_empty_base,
    )

    ley = [item for item in opportunities if item.company == "Ley Choon Group Holdings"]
    assert [item.role_title for item in ley] == [
        "Management Associate",
        "Engineer Associate",
        "EHS Officer",
        "Assistant Engineer",
        "Project Engineer",
    ]
    assert all(item.deadline_hint.isoformat() == "2026-11-12" for item in ley)

    point = next(item for item in opportunities if item.company.startswith("Point72"))
    assert point72 in point.urls
    assert point.deadline_hint is None

    avepoint = next(item for item in opportunities if item.role_title == "Backend Developer")
    assert avepoint.deadline_hint.isoformat() == "2026-08-31"
    assert errors == []
    assert metrics.llm_calls == 0


def test_goh_deadline_variants_with_spaced_ordinal() -> None:
    source_date = datetime(2026, 8, 27, tzinfo=timezone.utc)
    assert _parse_deadline("Deadline: 5 th Dec 2026", source_date).isoformat() == "2026-12-05"
    assert _parse_deadline("Deadline: 24th Dec 26", source_date).isoformat() == "2026-12-24"
    assert _parse_deadline("Deadline :25 Sep 26", source_date).isoformat() == "2026-09-25"
