from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_DB_PATH = Path("data/nusmods/nusmods_courses.sqlite3")
DEFAULT_ACADEMIC_YEARS = ("2026-2027", "2025-2026", "2024-2025", "2023-2024")
NUSMODS_BASE = "https://api.nusmods.com/v2"


@dataclass(frozen=True)
class StoredCourse:
    academic_year: str
    module_code: str
    title: str
    description: str
    department: str | None
    faculty: str | None
    module_credit: float | None
    semesters: tuple[int, ...]
    skills: tuple[str, ...]
    source_url: str


def connect_database(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS courses (
            academic_year TEXT NOT NULL,
            module_code TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            department TEXT,
            faculty TEXT,
            module_credit REAL,
            semesters_json TEXT NOT NULL DEFAULT '[]',
            skills_json TEXT NOT NULL DEFAULT '[]',
            source_url TEXT NOT NULL,
            PRIMARY KEY (academic_year, module_code)
        );

        CREATE INDEX IF NOT EXISTS idx_courses_module_code
            ON courses(module_code);
        CREATE INDEX IF NOT EXISTS idx_courses_title
            ON courses(title);
        CREATE INDEX IF NOT EXISTS idx_courses_faculty
            ON courses(faculty);

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    return connection


def _semesters(payload: dict) -> tuple[int, ...]:
    raw = payload.get("semesters")
    if isinstance(raw, list):
        values = []
        for item in raw:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value in {1, 2, 3, 4} and value not in values:
                values.append(value)
        if values:
            return tuple(sorted(values))

    values = []
    for semester in payload.get("semesterData") or []:
        try:
            value = int(semester.get("semester"))
        except (TypeError, ValueError, AttributeError):
            continue
        if value in {1, 2, 3, 4} and value not in values:
            values.append(value)
    return tuple(sorted(values))


def _module_credit(payload: dict) -> float | None:
    value = payload.get("moduleCredit")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_academic_year_modules(
    academic_year: str,
    *,
    timeout_seconds: float = 90.0,
) -> list[dict]:
    url = f"{NUSMODS_BASE}/{academic_year}/moduleInfo.json"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected NUSMods moduleInfo payload for {academic_year}")
    return payload


def build_database(
    *,
    academic_years: Iterable[str] = DEFAULT_ACADEMIC_YEARS,
    db_path: Path = DEFAULT_DB_PATH,
    refresh: bool = False,
) -> dict[str, int]:
    # Lazy import avoids a module-level circular dependency.
    from career_agent.course_enrichment import infer_course_skills

    connection = connect_database(db_path)
    counts: dict[str, int] = {}
    try:
        for academic_year in academic_years:
            if refresh:
                connection.execute(
                    "DELETE FROM courses WHERE academic_year = ?",
                    (academic_year,),
                )

            modules = fetch_academic_year_modules(academic_year)
            row_count = 0
            for payload in modules:
                module_code = str(payload.get("moduleCode") or "").strip().upper()
                title = str(payload.get("title") or "").strip()
                if not module_code or not title:
                    continue

                semesters = _semesters(payload)
                if not semesters:
                    continue

                description = str(payload.get("description") or "").strip()
                skills = tuple(sorted(infer_course_skills(title, description)))
                source_url = f"{NUSMODS_BASE}/{academic_year}/modules/{module_code}.json"

                connection.execute(
                    """
                    INSERT INTO courses (
                        academic_year, module_code, title, description,
                        department, faculty, module_credit,
                        semesters_json, skills_json, source_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(academic_year, module_code) DO UPDATE SET
                        title = excluded.title,
                        description = excluded.description,
                        department = excluded.department,
                        faculty = excluded.faculty,
                        module_credit = excluded.module_credit,
                        semesters_json = excluded.semesters_json,
                        skills_json = excluded.skills_json,
                        source_url = excluded.source_url
                    """,
                    (
                        academic_year,
                        module_code,
                        title,
                        description,
                        payload.get("department"),
                        payload.get("faculty"),
                        _module_credit(payload),
                        json.dumps(semesters),
                        json.dumps(skills),
                        source_url,
                    ),
                )
                row_count += 1

            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (f"academic_year:{academic_year}", str(row_count)),
            )
            connection.commit()
            counts[academic_year] = row_count
    finally:
        connection.close()
    return counts


def _row_to_course(row: sqlite3.Row) -> StoredCourse:
    return StoredCourse(
        academic_year=row["academic_year"],
        module_code=row["module_code"],
        title=row["title"],
        description=row["description"],
        department=row["department"],
        faculty=row["faculty"],
        module_credit=row["module_credit"],
        semesters=tuple(json.loads(row["semesters_json"] or "[]")),
        skills=tuple(json.loads(row["skills_json"] or "[]")),
        source_url=row["source_url"],
    )


def lookup_course(
    module_code: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> StoredCourse | None:
    if not db_path.exists():
        return None
    code = "".join((module_code or "").upper().split())
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            """
            SELECT * FROM courses
            WHERE module_code = ?
            ORDER BY academic_year DESC
            LIMIT 1
            """,
            (code,),
        ).fetchone()
        return _row_to_course(row) if row else None
    finally:
        connection.close()


def database_stats(db_path: Path = DEFAULT_DB_PATH) -> dict[str, object]:
    if not db_path.exists():
        return {"rows": 0, "unique_module_codes": 0, "academic_years": {}}
    connection = connect_database(db_path)
    try:
        rows = int(connection.execute("SELECT COUNT(*) FROM courses").fetchone()[0])
        unique_codes = int(
            connection.execute("SELECT COUNT(DISTINCT module_code) FROM courses").fetchone()[0]
        )
        by_year = {
            row[0]: int(row[1])
            for row in connection.execute(
                "SELECT academic_year, COUNT(*) FROM courses GROUP BY academic_year ORDER BY academic_year DESC"
            )
        }
        return {
            "rows": rows,
            "unique_module_codes": unique_codes,
            "academic_years": by_year,
        }
    finally:
        connection.close()
