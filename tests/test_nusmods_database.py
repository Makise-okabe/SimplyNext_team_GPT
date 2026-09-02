from __future__ import annotations

import json
from pathlib import Path

import career_agent.nusmods_database as db


def test_build_database_and_lookup_course(tmp_path, monkeypatch):
    database = tmp_path / "nusmods.sqlite3"

    monkeypatch.setattr(
        db,
        "fetch_academic_year_modules",
        lambda academic_year: [
            {
                "moduleCode": "EE2028",
                "title": "Microcontroller Programming and Interfacing",
                "description": "Embedded systems programming with microcontrollers, interfaces and C.",
                "department": "Electrical and Computer Engineering",
                "faculty": "College of Design and Engineering",
                "moduleCredit": "4",
                "semesters": [1, 2],
            },
            {
                "moduleCode": "CS3237",
                "title": "Introduction to Internet of Things",
                "description": "Internet of Things systems using embedded devices and software.",
                "department": "Computer Science",
                "faculty": "Computing",
                "moduleCredit": "4",
                "semesters": [1],
            },
        ],
    )

    counts = db.build_database(
        academic_years=("2026-2027",),
        db_path=database,
        refresh=True,
    )

    assert counts == {"2026-2027": 2}
    stats = db.database_stats(database)
    assert stats["rows"] == 2
    assert stats["unique_module_codes"] == 2

    ee2028 = db.lookup_course("EE2028", db_path=database)
    assert ee2028 is not None
    assert ee2028.title == "Microcontroller Programming and Interfacing"
    assert "embedded systems" in ee2028.skills
    assert ee2028.semesters == (1, 2)


def test_database_ignores_entries_without_semesters(tmp_path, monkeypatch):
    database = tmp_path / "nusmods.sqlite3"
    monkeypatch.setattr(
        db,
        "fetch_academic_year_modules",
        lambda academic_year: [
            {
                "moduleCode": "XX9999",
                "title": "Non Teaching Record",
                "description": "",
                "semesters": [],
            }
        ],
    )

    counts = db.build_database(
        academic_years=("2026-2027",),
        db_path=database,
        refresh=True,
    )
    assert counts == {"2026-2027": 0}
    assert db.database_stats(database)["rows"] == 0
