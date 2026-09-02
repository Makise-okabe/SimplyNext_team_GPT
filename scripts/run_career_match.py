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

from career_agent.stage1_ranking import rank_jobs
from career_agent.stage2_ranking import rerank_stage1
from career_agent.student_profile import build_student_profile

DEFAULT_JOBS = Path("data/job_records/latest_matching_candidates.json")
DEFAULT_PROFILE_OUTPUT = Path("data/student_profiles/latest_student_profile.json")
DEFAULT_STAGE1_OUTPUT = Path("data/matching/stage1_rankings.json")
DEFAULT_FINAL_OUTPUT = Path("data/matching/final_rankings.json")


def _pdf_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Export matching candidates before running the final matcher."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SimplyNext student-profile building, broad matching and semantic reranking."
    )
    parser.add_argument("--resume", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--stage1-top", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--profile-output", default=str(DEFAULT_PROFILE_OUTPUT))
    parser.add_argument("--stage1-output", default=str(DEFAULT_STAGE1_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_FINAL_OUTPUT))
    args = parser.parse_args()

    resume_path = Path(args.resume)
    transcript_path = Path(args.transcript)
    resume_text = _pdf_text(resume_path)
    transcript_text = _pdf_text(transcript_path)

    print("[1/3] Building student profile...")
    profile = build_student_profile(
        resume_text=resume_text,
        transcript_text=transcript_text,
        enrich_modules=True,
    )
    profile_payload = {
        "schema": "simplinext.student_profile.v1",
        "resume_file": resume_path.name,
        "transcript_file": transcript_path.name,
        **profile.to_dict(),
    }
    _write_json(Path(args.profile_output), profile_payload)
    print(
        f"      modules={len(profile.module_codes)} explicit_skills={len(profile.explicit_skills)} "
        f"course_skills={len(profile.course_derived_skills)} total_skills={len(profile.all_skills)}"
    )

    print("[2/3] Ranking active jobs deterministically...")
    jobs_payload = _load(Path(args.jobs))
    jobs = list(jobs_payload.get("jobs") or [])
    ranked = rank_jobs(profile_payload, jobs)
    stage1_payload = {
        "schema": "simplinext.stage1_ranking.v1",
        "student_profile": str(args.profile_output),
        "job_candidates": str(args.jobs),
        "job_count": len(ranked),
        "top_n": min(max(args.stage1_top, 0), len(ranked)),
        "rankings": [item.to_dict() for item in ranked],
    }
    _write_json(Path(args.stage1_output), stage1_payload)
    print(f"      ranked={len(ranked)} shortlist={stage1_payload['top_n']}")

    print("[3/3] Semantically reranking shortlist in small LLM batches...")
    final = rerank_stage1(
        resume_text=resume_text,
        student_profile=profile_payload,
        all_jobs=jobs,
        stage1_rankings=stage1_payload["rankings"],
        stage1_top_n=args.stage1_top,
    )
    semantic_assessed = sum(item.semantic_assessed for item in final)
    final_payload = {
        "schema": "simplinext.final_ranking.v2",
        "resume_file": resume_path.name,
        "transcript_file": transcript_path.name,
        "active_job_count": len(ranked),
        "stage1_shortlist_count": len(final),
        "semantic_assessed_count": semantic_assessed,
        "semantic_coverage": round(semantic_assessed / len(final), 4) if final else 0.0,
        "final_top_n": min(max(args.top, 0), len(final)),
        "rankings": [item.to_dict() for item in final],
    }
    _write_json(Path(args.output), final_payload)

    top_n = final_payload["final_top_n"]
    print()
    print("SIMPLYNEXT FINAL MATCHES")
    print("Active jobs      :", len(ranked))
    print("LLM shortlist    :", len(final))
    print(f"Semantic assessed: {semantic_assessed}/{len(final)}")
    print("Top shown        :", top_n)
    print("Output           :", args.output)
    print()

    for index, item in enumerate(final[:top_n], start=1):
        print(
            f"{index:02d}. {item.final_score:5.1f} | {item.fit_label:8s} | "
            f"{item.confidence:6s} | {item.company} — {item.title}"
        )
        print("    why     :", item.why_match)
        print("    evidence:", "; ".join(item.matched_evidence) if item.matched_evidence else "<none>")
        if item.missing_or_weak_evidence:
            print("    gaps    :", "; ".join(item.missing_or_weak_evidence))
        print("    job data:", item.evidence_level)
        print("    semantic:", "assessed" if item.semantic_assessed else "missing")
        print("    URL     :", item.application_url or item.official_job_url or "<unavailable>")


if __name__ == "__main__":
    main()
