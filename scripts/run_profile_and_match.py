from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

from career_agent.job_matching import rank_jobs
from career_agent.models.job_record import JobRecord
from career_agent.student_profile_extraction import extract_student_profile

DEFAULT_JOBS = Path("data/job_records/latest_job_records.json")
DEFAULT_PROFILE = Path("data/student_profile/latest_student_profile.json")
DEFAULT_MATCHES = Path("data/matches/latest_matches.json")


def _read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    raise ValueError(f"Unsupported file type: {path.suffix}. Use PDF or TXT for the MVP.")


def _load_jobs(path: Path) -> list[JobRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [JobRecord.model_validate(item) for item in payload.get("jobs", [])]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext: extract StudentProfile from resume/transcript and rank matching-ready jobs."
    )
    parser.add_argument("--resume", required=True, help="Resume PDF/TXT path")
    parser.add_argument("--transcript", default=None, help="Optional transcript PDF/TXT path")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS), help="Matching-ready JobRecord dataset")
    parser.add_argument("--profile-output", default=str(DEFAULT_PROFILE))
    parser.add_argument("--matches-output", default=str(DEFAULT_MATCHES))
    args = parser.parse_args()

    resume_path = Path(args.resume)
    transcript_path = Path(args.transcript) if args.transcript else None
    jobs_path = Path(args.jobs)

    resume_text = _read_text_file(resume_path)
    transcript_text = _read_text_file(transcript_path) if transcript_path else ""
    jobs = _load_jobs(jobs_path)
    if not jobs:
        raise RuntimeError(f"No matching-ready jobs found in {jobs_path}")

    print("=" * 112)
    print("SIMPLYNEXT — RESUME/TRANSCRIPT → STUDENT PROFILE → JOB MATCHING")
    print("=" * 112)
    print("Resume chars    :", len(resume_text))
    print("Transcript chars:", len(transcript_text))
    print("Jobs to match   :", len(jobs))

    profile = extract_student_profile(
        resume_text=resume_text,
        transcript_text=transcript_text,
    )
    ranked = rank_jobs(profile, jobs)

    profile_output = Path(args.profile_output)
    matches_output = Path(args.matches_output)
    _write_json(
        profile_output,
        {
            "schema": "simplinext.student_profile.v1",
            "profile": profile.model_dump(mode="json"),
        },
    )
    _write_json(
        matches_output,
        {
            "schema": "simplinext.match_results.v1",
            "match_count": len(ranked),
            "matches": [item.model_dump(mode="json") for item in ranked],
        },
    )

    print("\nTOP MATCHES")
    for index, result in enumerate(ranked[:10], start=1):
        print(
            f"{index:2}. {result.score:3}/100 | {result.recommendation:14} | "
            f"{result.company or '<unknown>'} — {result.title or '<unknown>'}"
        )
        if result.matched_strengths:
            print("    strengths:", "; ".join(result.matched_strengths[:3]))
        if result.missing_requirements:
            print("    gaps     :", "; ".join(result.missing_requirements[:3]))
        print("    rationale:", " ".join(result.rationale.split())[:260])
        if result.jd_source_url:
            print("    JD source:", result.jd_source_url)

    print("\nStudentProfile:", profile_output)
    print("Match results  :", matches_output)


if __name__ == "__main__":
    main()
