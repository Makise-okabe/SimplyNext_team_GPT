from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

# The matching run may return perfectly valid Unicode from the LLM (for example
# non-breaking hyphens). On Windows, the console can still default to GBK/cp936.
# Keep the native console encoding, but replace characters it cannot represent
# instead of crashing after the result JSON has already been written.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.jd_enricher import enrich_job_description
from career_agent.job_link_resolver import resolve_job_link
from career_agent.matching_dataset import matching_evidence_level, matching_input_text
from career_agent.models.job_record import JobRecord
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


def _key(company: str | None, title: str | None) -> tuple[str, str]:
    return (
        " ".join(str(company or "").lower().split()),
        " ".join(str(title or "").lower().split()),
    )


def _job_lookup(jobs: list[dict]) -> dict[tuple[str, str], dict]:
    return {_key(job.get("company"), job.get("title")): job for job in jobs}


def _record_from_payload(payload: dict) -> JobRecord:
    allowed = set(JobRecord.model_fields)
    return JobRecord.model_validate({key: value for key, value in payload.items() if key in allowed})


def _matching_payload(record: JobRecord, original: dict) -> dict:
    payload = dict(original)
    payload.update(record.model_dump(mode="json"))
    payload["matching_evidence_level"] = matching_evidence_level(record)
    payload["matching_input_text"] = matching_input_text(record)
    return payload


def _deep_rescue_ranked_jobs(
    *,
    jobs: list[dict],
    rankings: list[dict],
    top_n: int,
    label: str,
) -> tuple[list[dict], dict[str, int]]:
    """Do a bounded second-pass search only for the shortlist shown to the user.

    The broad agent deliberately keeps web work cheap. This pass improves recall
    for the final CTA without sending all 137 jobs back through deeper search.
    It also re-fetches the JD when a newly recovered exact page is found, so the
    Stage-2 semantic judge can use that evidence.
    """
    lookup = _job_lookup(jobs)
    updated_by_key: dict[tuple[str, str], dict] = {}
    attempted = 0
    resolved = 0
    full_jd = 0
    partial_jd = 0

    for ranked in rankings[: max(top_n, 0)]:
        key = _key(ranked.get("company"), ranked.get("title"))
        raw = lookup.get(key)
        if not raw:
            continue

        existing_kind = str(raw.get("job_page_kind") or "unresolved")
        existing_jd = str(raw.get("jd_status") or "unavailable")
        if existing_kind in {"official_exact", "secondary_exact"} and existing_jd in {
            "fetched_official",
            "fetched_secondary",
            "partial_official",
            "partial_secondary",
        }:
            continue

        attempted += 1
        print(
            f"      [{label} {attempted:02}] {raw.get('company')} — {raw.get('title')}"
        )
        record = _record_from_payload(raw)
        rescued, resolution = resolve_job_link(record, deep_search=True)
        enriched = enrich_job_description(rescued)
        payload = _matching_payload(enriched, raw)
        updated_by_key[key] = payload

        if resolution.url:
            resolved += 1
            print(f"          exact-link rescue -> {resolution.kind} | {resolution.url}")
        else:
            print("          exact-link rescue -> unresolved")

        if enriched.jd_status in {"fetched_official", "fetched_secondary"}:
            full_jd += 1
            print(f"          rescued JD -> full ({enriched.jd_status})")
        elif enriched.jd_status in {"partial_official", "partial_secondary"}:
            partial_jd += 1
            print(f"          rescued JD -> partial ({enriched.jd_status})")

    updated_jobs = [
        updated_by_key.get(_key(job.get("company"), job.get("title")), job)
        for job in jobs
    ]
    return updated_jobs, {
        "attempted": attempted,
        "resolved": resolved,
        "full_jd": full_jd,
        "partial_jd": partial_jd,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the SimplyNext Career Opportunity Agent v2.")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--transcript", required=True)
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
    print("SIMPLYNEXT SEARCH CONFIG")
    print(f"Stable Search API : {provider or 'NOT configured'}")
    print("Fallback Search   : Bing/DDG public scraping")
    print()

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
    print(
        f"      modules={len(profile.module_codes)} explicit_skills={len(profile.explicit_skills)} "
        f"course_skills={len(profile.course_derived_skills)} total_skills={len(profile.all_skills)}"
    )

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
    print(
        "      web summary: "
        f"selected={m.web_selected} links={m.links_resolved}/{m.web_selected} "
        f"official={m.official_links} secondary={m.secondary_links} "
        f"unresolved={m.unresolved_links}"
    )
    print(f"      JD summary : full={m.full_jd} partial={m.partial_jd}")

    # Fast broad search can miss a real posting because a provider returns only
    # a few nearby results. Rescue only the semantic shortlist with stricter,
    # cleaned-title queries before Stage 2 sees the evidence.
    enriched_jobs, rescue_metrics = _deep_rescue_ranked_jobs(
        jobs=agent.jobs,
        rankings=agent.stage1_rankings,
        top_n=args.semantic_top,
        label="JD RESCUE",
    )
    print(
        "      shortlist rescue: "
        f"attempted={rescue_metrics['attempted']} resolved={rescue_metrics['resolved']} "
        f"full_jd={rescue_metrics['full_jd']} partial_jd={rescue_metrics['partial_jd']}"
    )

    fallback_only = sum(
        1 for job in enriched_jobs if job.get("search_resolution_status") == "search_fallback_only"
    )

    print("[3/4] Semantically reranking the best email opportunities...")
    final = rerank_stage1(
        resume_text=resume_text,
        student_profile=profile_payload,
        all_jobs=enriched_jobs,
        stage1_rankings=agent.stage1_rankings,
        stage1_top_n=args.semantic_top,
        show_progress=True,
    )
    semantic_assessed = sum(item.semantic_assessed for item in final)

    print("[4/4] Preparing related-role recommendations...")
    related_jobs, related_rescue = _deep_rescue_ranked_jobs(
        jobs=agent.related_jobs,
        rankings=agent.related_rankings,
        top_n=min(6, len(agent.related_rankings)),
        label="RELATED RESCUE",
    )
    print(f"      related jobs discovered={m.related_jobs_discovered}")

    lookup = _job_lookup(enriched_jobs)
    top_n = min(max(args.top, 0), len(final))
    final_cards = []
    for item in final[:top_n]:
        key = _key(item.company, item.title)
        job = lookup.get(key, {})
        final_cards.append(
            {
                **item.to_dict(),
                "job_page_url": (
                    job.get("jd_source_url")
                    or job.get("job_page_url")
                    or item.application_url
                    or item.official_job_url
                ),
                "job_page_kind": job.get("job_page_kind", "unresolved"),
                "job_page_confidence": job.get("job_page_confidence", "low"),
                "search_fallback_url": job.get("search_fallback_url"),
                "search_resolution_status": job.get("search_resolution_status", "not_searched"),
                "jd_status": job.get("jd_status", "unavailable"),
                "jd_source_url": job.get("jd_source_url"),
                "source_key": job.get("source_key"),
            }
        )

    related_lookup = _job_lookup(related_jobs)
    related_cards = []
    main_keys = {_key(card.get("company"), card.get("title")) for card in final_cards}
    for item in agent.related_rankings[:8]:
        key = _key(item.get("company"), item.get("title"))
        if key in main_keys:
            continue
        job = related_lookup.get(key, {})
        related_cards.append(
            {
                **item,
                "job_page_url": (
                    job.get("jd_source_url")
                    or job.get("job_page_url")
                    or job.get("application_url")
                ),
                "job_page_kind": job.get("job_page_kind", "unresolved"),
                "job_page_confidence": job.get("job_page_confidence", "low"),
                "jd_status": job.get("jd_status", "unavailable"),
                "jd_source_url": job.get("jd_source_url"),
                "recommendation_reason": "Related role discovered from a company already matching the student well.",
            }
        )

    output = {
        "schema": "simplinext.career_opportunity_agent.v2",
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
            "top_link_rescue_attempted": rescue_metrics["attempted"],
            "top_link_rescue_resolved": rescue_metrics["resolved"],
            "top_link_rescue_full_jd": rescue_metrics["full_jd"],
            "top_link_rescue_partial_jd": rescue_metrics["partial_jd"],
            "related_link_rescue_resolved": related_rescue["resolved"],
        },
        "top_matches": final_cards,
        "related_jobs": related_cards,
        "all_rankings": agent.stage1_rankings,
    }
    _write(Path(args.output), output)

    print()
    print("SIMPLYNEXT CAREER OPPORTUNITY AGENT")
    print(f"Email jobs          : {m.active_jobs}")
    print(f"Web-enriched jobs   : {m.web_selected}")
    print(f"Initial job links   : {m.links_resolved}/{m.web_selected}")
    print(f"Top link rescues    : {rescue_metrics['resolved']}/{rescue_metrics['attempted']}")
    print(f"Search fallbacks    : {fallback_only}")
    print(f"Initial full/partial: {m.full_jd} / {m.partial_jd}")
    print(f"Rescued full/partial: {rescue_metrics['full_jd']} / {rescue_metrics['partial_jd']}")
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
        else:
            print("    View Job : <unresolved>")

    if related_cards:
        print()
        print("YOU MAY ALSO LIKE")
        for card in related_cards[:6]:
            print(f"- {card['company']} — {card['title']} | score={card['score']}")
            print(f"  URL: {card['job_page_url'] or '<unresolved>'}")


if __name__ == "__main__":
    main()
