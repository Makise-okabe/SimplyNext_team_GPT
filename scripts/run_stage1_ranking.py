from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.stage1_ranking import rank_jobs

DEFAULT_STUDENT = Path("data/student_profiles/latest_student_profile.json")
DEFAULT_JOBS = Path("data/job_records/latest_matching_candidates.json")
DEFAULT_OUTPUT = Path("data/matching/stage1_rankings.json")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank all active jobs against a student profile without LLM calls.")
    parser.add_argument("--student", default=str(DEFAULT_STUDENT))
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    student_payload = _load(Path(args.student))
    jobs_payload = _load(Path(args.jobs))
    jobs = list(jobs_payload.get("jobs") or [])
    ranked = rank_jobs(student_payload, jobs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "simplinext.stage1_ranking.v1",
        "student_profile": str(args.student),
        "job_candidates": str(args.jobs),
        "job_count": len(ranked),
        "top_n": min(max(args.top, 0), len(ranked)),
        "rankings": [item.to_dict() for item in ranked],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top_n = min(max(args.top, 0), len(ranked))
    print("STAGE 1 MATCHING SUMMARY")
    print("Jobs ranked :", len(ranked))
    print("Top shown   :", top_n)
    print("Output      :", output)
    print()
    for index, item in enumerate(ranked[:top_n], start=1):
        matched = list(item.matched_resume_skills) + list(item.matched_course_skills)
        print(f"{index:02d}. {item.score:5.1f} | {item.confidence:6s} | {item.company} — {item.title}")
        print("    matched:", ", ".join(matched) if matched else "<none>")
        print("    evidence:", item.evidence_level)


if __name__ == "__main__":
    main()
