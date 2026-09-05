from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.presentation import job_card
from career_agent.record_identity import record_key
from career_agent.opportunity_agent import run_opportunity_agent
from career_agent.stage2_ranking import rerank_stage1
from career_agent.student_profile import build_student_profile
from career_agent.tools.web_search import stable_search_api_name

DEFAULT_JOBS = Path("data/job_records/latest_matching_candidates.json")
DEFAULT_OUTPUT = Path("data/matching/career_opportunity_agent.json")


def _pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _job_lookup(jobs: list[dict]) -> dict[str, dict]:
    return {record_key(job): job for job in jobs}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the SimplyNext Career Opportunity Agent v2.")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--profile", help="Reuse the student profile already built by the UI")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--web-primary", type=int, default=12)
    parser.add_argument("--web-explore", type=int, default=3)
    parser.add_argument(
        "--semantic-top",
        type=int,
        default=5,
        help="Number of jobs sent to the final LLM semantic review. Default 5 keeps the demo to one batch.",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    provider = stable_search_api_name()
    print(f"[SEARCH] Provider: {provider or 'NOT configured (Bing/DDG fallback only)'}")
    print()

    resume_path = Path(args.resume)
    transcript_path = Path(args.transcript)
    resume_text = _pdf_text(resume_path)
    transcript_text = _pdf_text(transcript_path)

    print("[1/4] Building student profile...")
    if args.profile:
        profile_payload = _load(Path(args.profile))
    else:
        profile = build_student_profile(resume_text=resume_text, transcript_text=transcript_text, enrich_modules=True)
        profile_payload = {"schema": "simplinext.student_profile.v1", "resume_file": resume_path.name, "transcript_file": transcript_path.name, **profile.to_dict()}
    print(f"      modules={len(profile_payload.get('module_codes', []))} skills={len(profile_payload.get('all_skills', []))}")

    print("[2/4] Rough-ranking all email jobs, then enriching only the strongest shortlist...")
    jobs = list((_load(Path(args.jobs)).get("jobs") or []))
    agent = run_opportunity_agent(
        student_profile=profile_payload,
        jobs=jobs,
        web_primary_count=args.web_primary,
        web_exploration_count=args.web_explore,
        semantic_shortlist_count=args.semantic_top,
        progress=print,
    )
    m = agent.metrics
    fallback_only = sum(
        1 for job in agent.jobs if job.get("search_resolution_status") == "search_fallback_only"
    )
    print(
        "      web summary: "
        f"selected={m.web_selected} links={m.links_resolved}/{m.web_selected} "
        f"official={m.official_links} secondary={m.secondary_links} "
        f"fallback={fallback_only} unresolved={m.unresolved_links} "
        f"searches={m.search_calls} pages_checked={m.page_fetch_calls}"
    )
    print(f"      JD summary : full={m.full_jd} partial={m.partial_jd}")

    print("[3/4] Semantically reranking the best email opportunities...")
    final = rerank_stage1(
        resume_text=resume_text,
        student_profile=profile_payload,
        all_jobs=agent.jobs,
        stage1_rankings=agent.stage1_rankings,
        stage1_top_n=args.semantic_top,
        show_progress=True,
    )
    semantic_assessed = sum(item.semantic_assessed for item in final)

    print("[4/4] Preparing related-role recommendations...")
    print(f"      related jobs discovered={m.related_jobs_discovered}")

    lookup = _job_lookup(agent.jobs)
    top_n = min(max(args.top, 0), len(final))
    final_cards = [job_card(lookup.get(item.record_id, {}), item.to_dict()) for item in final[:top_n]]
    related_lookup = _job_lookup(agent.related_jobs)
    main_keys = {card.get("record_id") for card in final_cards}
    related_cards = []
    for item in agent.related_rankings[:8]:
        key = record_key(item)
        if key in main_keys:
            continue
        card = job_card(related_lookup.get(key, {}), item)
        card["recommendation_reason"] = "Another role from the same company that may fit your background."
        related_cards.append(card)
    all_cards = [job_card(lookup.get(record_key(item), {}), item) for item in agent.stage1_rankings]

    output = {
        "schema": "simplinext.career_opportunity_agent.v3",
        "resume_file": resume_path.name,
        "transcript_file": transcript_path.name,
        "search": {
            "stable_api": provider,
            "fallback": "bing_ddg_public_scraping",
        },
        "metrics": {
            **m.__dict__,
            "search_fallback_only": fallback_only,
            "semantic_assessed": semantic_assessed,
            "semantic_shortlist": len(final),
        },
        "top_matches": final_cards,
        "related_jobs": related_cards,
        "all_rankings": all_cards,
    }
    _write(Path(args.output), output)

    print()
    print("SIMPLYNEXT CAREER OPPORTUNITY AGENT")
    print(f"Email jobs          : {m.active_jobs}")
    print(f"Web-enriched jobs   : {m.web_selected}")
    print(f"Resolved job links  : {m.links_resolved}/{m.web_selected}")
    print(f"Search fallbacks    : {fallback_only}")
    print(f"Full / partial JDs  : {m.full_jd} / {m.partial_jd}")
    print(f"Semantic assessed   : {semantic_assessed}/{len(final)}")
    print(f"Related discoveries : {m.related_jobs_discovered}")
    print(f"Output              : {args.output}")
    print()

    for index, card in enumerate(final_cards, start=1):
        print(f"{index:02d}. {card['final_score']:5.1f} | {card['company']} — {card['title']}")
        print(f"    match    : {card['why_match']}")
        print(f"    evidence : {card['evidence_level']} / {card['jd_status']}")
        print(f"    page     : {card['job_page_kind']} / {card['job_page_confidence']}")
        if card["job_page_url"]:
            print(f"    View Job : {card['job_page_url']}")
        elif card["search_fallback_url"]:
            print(f"    Find Job : {card['search_fallback_url']}")
        else:
            print("    Find Job : <unavailable>")

    if related_cards:
        print()
        print("YOU MAY ALSO LIKE")
        for card in related_cards[:6]:
            print(f"- {card['company']} — {card['title']} | score={card['score']}")
            print(f"  URL: {card['job_page_url'] or '<unresolved>'}")


if __name__ == "__main__":
    main()
