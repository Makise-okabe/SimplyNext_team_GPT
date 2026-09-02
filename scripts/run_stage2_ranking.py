from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.stage2_ranking import rerank_stage1

DEFAULT_STUDENT = Path("data/student_profiles/latest_student_profile.json")
DEFAULT_JOBS = Path("data/job_records/latest_matching_candidates.json")
DEFAULT_STAGE1 = Path("data/matching/stage1_rankings.json")
DEFAULT_OUTPUT = Path("data/matching/final_rankings.json")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _pdf_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantically rerank the Stage-1 shortlist with one LLM call.")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--student", default=str(DEFAULT_STUDENT))
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--stage1", default=str(DEFAULT_STAGE1))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stage1-top", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    student = _load(Path(args.student))
    jobs_payload = _load(Path(args.jobs))
    stage1_payload = _load(Path(args.stage1))
    resume_text = _pdf_text(Path(args.resume))

    results = rerank_stage1(
        resume_text=resume_text,
        student_profile=student,
        all_jobs=list(jobs_payload.get("jobs") or []),
        stage1_rankings=list(stage1_payload.get("rankings") or []),
        stage1_top_n=args.stage1_top,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "simplinext.final_ranking.v1",
        "stage1_shortlist_count": len(results),
        "final_top_n": min(max(args.top, 0), len(results)),
        "rankings": [item.to_dict() for item in results],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top_n = min(max(args.top, 0), len(results))
    print("FINAL SEMANTIC MATCHING SUMMARY")
    print("Stage-1 candidates reranked:", len(results))
    print("Top shown                  :", top_n)
    print("Output                     :", output)
    print()

    for index, item in enumerate(results[:top_n], start=1):
        print(
            f"{index:02d}. {item.final_score:5.1f} | {item.fit_label:8s} | "
            f"{item.confidence:6s} | {item.company} — {item.title}"
        )
        print("    why     :", item.why_match)
        print("    evidence:", "; ".join(item.matched_evidence) if item.matched_evidence else "<none>")
        if item.missing_or_weak_evidence:
            print("    gaps    :", "; ".join(item.missing_or_weak_evidence))
        print("    job data:", item.evidence_level)
        print("    URL     :", item.application_url or item.official_job_url or "<unavailable>")


if __name__ == "__main__":
    main()
