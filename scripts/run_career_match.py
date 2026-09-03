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

from career_agent.shortlist_web_enrichment import enrich_stage1_shortlist
from career_agent.stage1_ranking import rank_jobs
from career_agent.stage2_ranking import rerank_stage1
from career_agent.student_profile import build_student_profile

DEFAULT_JOBS = Path("data/job_records/latest_matching_candidates.json")
DEFAULT_PROFILE_OUTPUT = Path("data/student_profiles/latest_student_profile.json")
DEFAULT_STAGE1_OUTPUT = Path("data/matching/stage1_rankings.json")
DEFAULT_ENRICHED_OUTPUT = Path("data/matching/enriched_matching_candidates.json")
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
        description=(
            "Run SimplyNext profile building, broad deterministic matching, "
            "official-web shortlist enrichment, and semantic reranking."
        )
    )
    parser.add_argument("--resume", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--stage1-top", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--skip-web-enrichment", action="store_true")
    parser.add_argument("--profile-output", default=str(DEFAULT_PROFILE_OUTPUT))
    parser.add_argument("--stage1-output", default=str(DEFAULT_STAGE1_OUTPUT))
    parser.add_argument("--enriched-output", default=str(DEFAULT_ENRICHED_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_FINAL_OUTPUT))
    args = parser.parse_args()

    resume_path = Path(args.resume)
    transcript_path = Path(args.transcript)
    resume_text = _pdf_text(resume_path)
    transcript_text = _pdf_text(transcript_path)

    print("[1/4] Building student profile...")
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

    print("[2/4] Ranking active jobs deterministically...")
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

    enriched_jobs = jobs
    enrichment_metrics = None
    if args.skip_web_enrichment:
        print("[3/4] Skipping official-web shortlist enrichment...")
    else:
        print("[3/4] Enriching Stage-1 shortlist from official web...")
        enriched_jobs, enrichment_metrics = enrich_stage1_shortlist(
            all_jobs=jobs,
            stage1_rankings=stage1_payload["rankings"],
            stage1_top_n=args.stage1_top,
            progress=print,
        )
        enriched_payload = {
            "schema": "simplinext.enriched_matching_candidates.v1",
            "source": str(args.jobs),
            "selected": enrichment_metrics.selected,
            "already_full_jd": enrichment_metrics.already_full_jd,
            "researched": enrichment_metrics.researched,
            "upgraded_to_full_jd": enrichment_metrics.upgraded_to_full_jd,
            "closed_by_official": enrichment_metrics.closed_by_official,
            "still_source_only": enrichment_metrics.still_source_only,
            "web_search_calls": enrichment_metrics.search_calls,
            "page_fetch_calls": enrichment_metrics.fetch_calls,
            "jobs": enriched_jobs,
        }
        _write_json(Path(args.enriched_output), enriched_payload)
        print(
            "      web summary: "
            f"selected={enrichment_metrics.selected} "
            f"already_full_jd={enrichment_metrics.already_full_jd} "
            f"upgraded={enrichment_metrics.upgraded_to_full_jd} "
            f"closed={enrichment_metrics.closed_by_official} "
            f"source_only={enrichment_metrics.still_source_only}"
        )
        print(
            f"      web calls: searches={enrichment_metrics.search_calls} "
            f"fetches={enrichment_metrics.fetch_calls}"
        )

        # Re-run deterministic ranking because full-JD evidence can change skill
        # extraction, evidence confidence, and shortlist ordering.
        reranked = rank_jobs(profile_payload, enriched_jobs)
        stage1_payload = {
            **stage1_payload,
            "job_candidates": str(args.enriched_output),
            "rankings": [item.to_dict() for item in reranked],
            "top_n": min(max(args.stage1_top, 0), len(reranked)),
        }
        ranked = reranked
        _write_json(Path(args.stage1_output), stage1_payload)

    print("[4/4] Semantically reranking shortlist in small LLM batches...")
    final = rerank_stage1(
        resume_text=resume_text,
        student_profile=profile_payload,
        all_jobs=enriched_jobs,
        stage1_rankings=stage1_payload["rankings"],
        stage1_top_n=args.stage1_top,
        show_progress=True,
    )
    semantic_assessed = sum(item.semantic_assessed for item in final)
    final_payload = {
        "schema": "simplinext.final_ranking.v3",
        "resume_file": resume_path.name,
        "transcript_file": transcript_path.name,
        "active_job_count": len(ranked),
        "stage1_shortlist_count": len(final),
        "semantic_assessed_count": semantic_assessed,
        "semantic_coverage": round(semantic_assessed / len(final), 4) if final else 0.0,
        "web_enrichment": (
            None
            if enrichment_metrics is None
            else {
                "selected": enrichment_metrics.selected,
                "already_full_jd": enrichment_metrics.already_full_jd,
                "researched": enrichment_metrics.researched,
                "upgraded_to_full_jd": enrichment_metrics.upgraded_to_full_jd,
                "closed_by_official": enrichment_metrics.closed_by_official,
                "still_source_only": enrichment_metrics.still_source_only,
                "search_calls": enrichment_metrics.search_calls,
                "fetch_calls": enrichment_metrics.fetch_calls,
            }
        ),
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
    if enrichment_metrics is not None:
        print(
            "Full-JD upgrades :",
            f"{enrichment_metrics.upgraded_to_full_jd} "
            f"(+{enrichment_metrics.already_full_jd} already available)",
        )
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
