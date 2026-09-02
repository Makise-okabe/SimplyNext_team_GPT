from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests

from career_agent.hybrid_matching import extract_explicit_skills, normalize_skills

DEFAULT_ACAD_YEARS = ("2026-2027", "2025-2026", "2024-2025", "2023-2024")
DEFAULT_CACHE = Path("data/course_skill_cache.json")
MODULE_CODE_PATTERN = re.compile(r"\b([A-Z]{2,5})\s*(\d{4}[A-Z]{0,2})\b")

COURSE_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"programming methodology", re.I), ("software engineering",)),
    (re.compile(r"data structures? and algorithms?", re.I), ("software engineering", "data analysis")),
    (re.compile(r"microcontroller|embedded", re.I), ("embedded systems", "c/c++", "digital design")),
    (re.compile(r"machine learning", re.I), ("machine learning", "python", "data analysis")),
    (re.compile(r"artificial intelligence|ai for design", re.I), ("machine learning", "python")),
    (re.compile(r"internet of things", re.I), ("iot", "embedded systems", "software engineering")),
    (re.compile(r"signals? and systems?|signal processing", re.I), ("signal processing",)),
    (re.compile(r"electronic circuits?|analog", re.I), ("analog circuits",)),
    (re.compile(r"microelectronic|semiconductor|devices? & sensors?|devices and sensors", re.I), ("semiconductor",)),
    (re.compile(r"photonics", re.I), ("photonics",)),
    (re.compile(r"electromagnetics", re.I), ("rf/wireless",)),
    (re.compile(r"probability|statistics", re.I), ("quantitative analysis", "data analysis")),
    (re.compile(r"project management", re.I), ("project management", "stakeholder management")),
    (re.compile(r"quantitative reasoning", re.I), ("data analysis", "quantitative analysis")),
    (re.compile(r"electrical energy systems", re.I), ("power systems",)),
    (re.compile(r"design thinking|design and make", re.I), ("stakeholder management", "communication")),
)


@dataclass(frozen=True)
class CourseEnrichment:
    module_code: str
    title: str
    description: str
    academic_year: str | None
    source_url: str | None
    skills: tuple[str, ...]
    source_status: str


def normalize_module_code(value: str) -> str:
    return re.sub(r"\s+", "", value or "").upper()


def extract_module_codes(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in MODULE_CODE_PATTERN.finditer(text or ""):
        prefix, suffix = match.groups()
        if prefix == "YEAR":
            continue
        code = normalize_module_code(prefix + suffix)
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def infer_course_skills(title: str, description: str = "") -> set[str]:
    text = f"{title}\n{description}".strip()
    skills = extract_explicit_skills(text)
    for pattern, inferred in COURSE_RULES:
        if pattern.search(text):
            skills.update(normalize_skills(inferred))
    return skills


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _from_local_database(module_code: str) -> CourseEnrichment | None:
    from career_agent.nusmods_database import lookup_course

    stored = lookup_course(module_code)
    if stored is None:
        return None
    return CourseEnrichment(
        module_code=stored.module_code,
        title=stored.title,
        description=stored.description,
        academic_year=stored.academic_year,
        source_url=stored.source_url,
        skills=stored.skills,
        source_status="local_nusmods_db",
    )


def fetch_nusmods_course(
    module_code: str,
    *,
    academic_years: Iterable[str] = DEFAULT_ACAD_YEARS,
    timeout_seconds: float = 5.0,
) -> CourseEnrichment:
    code = normalize_module_code(module_code)
    for year in academic_years:
        url = f"https://api.nusmods.com/v2/{year}/modules/{code}.json"
        try:
            response = requests.get(url, timeout=timeout_seconds)
        except requests.RequestException:
            continue
        if response.status_code == 404:
            continue
        try:
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        title = str(payload.get("title") or code).strip()
        description = str(payload.get("description") or "").strip()
        skills = infer_course_skills(title, description)
        return CourseEnrichment(
            module_code=code,
            title=title,
            description=description,
            academic_year=year,
            source_url=url,
            skills=tuple(sorted(skills)),
            source_status="fetched_nusmods",
        )

    return CourseEnrichment(
        module_code=code,
        title=code,
        description="",
        academic_year=None,
        source_url=None,
        skills=(),
        source_status="unavailable",
    )


def enrich_courses(
    module_codes: Iterable[str],
    *,
    cache_path: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> list[CourseEnrichment]:
    cache = _load_cache(cache_path)
    results: list[CourseEnrichment] = []

    for raw_code in module_codes:
        code = normalize_module_code(raw_code)

        if not refresh:
            local = _from_local_database(code)
            if local is not None:
                results.append(local)
                continue

        cached = cache.get(code)
        if cached and not refresh:
            results.append(
                CourseEnrichment(
                    module_code=code,
                    title=str(cached.get("title") or code),
                    description=str(cached.get("description") or ""),
                    academic_year=cached.get("academic_year"),
                    source_url=cached.get("source_url"),
                    skills=tuple(cached.get("skills") or ()),
                    source_status=str(cached.get("source_status") or "cached"),
                )
            )
            continue

        item = fetch_nusmods_course(code)
        results.append(item)
        cache[code] = asdict(item)

    _save_cache(cache_path, cache)
    return results
