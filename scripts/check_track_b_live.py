from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.batch_sources import build_source_corpus
from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.connectors.outlook_graph import CAREER_SOURCE_BY_SENDER, OutlookGraphConnector
from career_agent.job_catalog_pipeline import (
    _canonical_company_text,
    _extract,
    _normalize_extracted_signals,
)
from career_agent.models.inbox import CareerEmailRecord

DEFAULT_COMPANIES = ["P&G", "Reolink", "Tesla", "Point72"]
DEFAULT_OUTPUT = Path("data/job_records/track_b_live_check.json")


def _source_key(sender_email: str | None) -> str | None:
    return CAREER_SOURCE_BY_SENDER.get((sender_email or "").strip().lower())


def _matches_company(company: str | None, filters: list[str]) -> bool:
    canonical = _canonical_company_text(company)
    aliases = {_canonical_company_text(value) for value in filters}
    return any(
        alias == canonical or alias in canonical or canonical in alias
        for alias in aliases
    )


def _corpus_aliases(company: str) -> list[str]:
    canonical = _canonical_company_text(company)
    if canonical == "procter gamble":
        return ["p&g", "p and g", "procter & gamble", "procter and gamble", "pgcareers"]
    if canonical == "point72":
        return ["point72"]
    return [token for token in re.findall(r"[a-z0-9]+", company.lower()) if len(token) >= 3]


def _corpus_hits(corpus: str, company: str, limit: int = 12) -> list[str]:
    aliases = _corpus_aliases(company)
    hits: list[str] = []
    for raw_line in corpus.splitlines():
        line = " ".join(raw_line.split()).strip()
        lowered = line.lower()
        if line and any(alias in lowered for alias in aliases):
            hits.append(line)
            if len(hits) >= limit:
                break
    return hits


def _short(value: str | None, limit: int = 180) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused live Track B sanity check for a few Goh companies."
    )
    parser.add_argument("--scan", type=int, default=30)
    parser.add_argument(
        "--company",
        action="append",
        dest="companies",
        help=(
            "Company to check. Repeat flag for multiple companies. "
            "Defaults to P&G, Reolink, Tesla, Point72."
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    companies = args.companies or DEFAULT_COMPANIES
    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    goh = next(
        (
            message
            for message in messages
            if _source_key(message.sender_email) == "goh_ze_li"
        ),
        None,
    )
    if goh is None:
        raise RuntimeError("No trusted Goh Ze Li email found in scan window.")

    record = CareerEmailRecord(source="goh_ze_li", email=goh)
    corpus, _, _, source_warnings = build_source_corpus(
        goh,
        fetch_linked_pdfs=True,
    )
    opportunities, metrics, extraction_errors = _extract(record, corpus)
    opportunities = _normalize_extracted_signals(opportunities)

    selected = [
        signal
        for signal in opportunities
        if _matches_company(signal.company, companies)
    ]
    if not selected:
        raise RuntimeError(f"No opportunities matched companies: {companies}")

    groups: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for index, signal in enumerate(selected, start=1):
        groups[_canonical_company_text(signal.company)].append((index, signal))

    print("=" * 104)
    print("SIMPLYNEXT TRACK B — FOCUSED LIVE CHECK")
    print("=" * 104)
    print("Companies requested:", ", ".join(companies))
    print("Companies matched  :", len(groups))
    print("Roles matched      :", len(selected))
    print("Extraction LLM     :", metrics.llm_calls)
    print("Source subject     :", _short(goh.subject, 200))
    print("Source received    :", getattr(goh, "received_at", None))
    print("Source message id  :", _short(goh.message_id, 120))
    print("Corpus chars       :", len(corpus))
    print("\nExpected behavior: source link first → official exact role → LinkedIn fallback")

    missing = [
        requested
        for requested in companies
        if not any(_matches_company(signal.company, [requested]) for signal in opportunities)
    ]
    if missing:
        print("Companies absent after production normalization:", ", ".join(missing))
        print("\nMISSING COMPANY SOURCE EVIDENCE")
        for requested in missing:
            hits = _corpus_hits(corpus, requested)
            print(f"  {requested}: corpus_hits={len(hits)}")
            for line in hits:
                print("    ", _short(line, 240))
            if not hits:
                print("     <no matching company alias found in current corpus>")

    context = ResearchContext()
    all_jobs = []
    for company_index, (_, items) in enumerate(groups.items(), start=1):
        company = items[0][1].company or "<unknown>"
        print("\n" + "-" * 104)
        print(
            f"COMPANY {company_index}/{len(groups)} | "
            f"{company} | {len(items)} role(s)"
        )
        for _, signal in items:
            print(f"  SOURCE | {signal.role_title}")
            for url in signal.urls:
                print(f"           {url}")

        outcome = research_company_jobs(
            email=goh,
            source_key="goh_ze_li",
            company_items=items,
            context=context,
            progress=print,
        )
        all_jobs.extend(outcome.job_records)

        for job in outcome.job_records:
            print(f"\n  RESULT | {job.title}")
            print("    jd_status      :", job.jd_status)
            print("    primary        :", job.primary_source_url)
            print("    secondary      :", job.secondary_source_url)
            print("    application    :", job.application_url)
            print("    jd_source      :", job.jd_source_url)
            if job.warnings:
                print("    warnings       :", " | ".join(job.warnings[:3]))

    print("\n" + "=" * 104)
    print("SEARCH TRACE — ACTUAL RETURNED RESULTS")
    print("=" * 104)
    for query, results in context.search_cache.items():
        print("QUERY:", _short(query, 240))
        print("  usable results:", len(results))
        if not results:
            print("   <none>")
            continue
        for index, result in enumerate(results[:3], start=1):
            print(f"  {index}. {_short(result.title, 160)}")
            print("     ", result.url)

    payload = {
        "schema": "simplinext.track_b.live_check.v2",
        "companies_requested": companies,
        "source_subject": goh.subject,
        "source_received_at": str(getattr(goh, "received_at", None)),
        "source_message_id": goh.message_id,
        "missing_company_corpus_hits": {
            company: _corpus_hits(corpus, company) for company in missing
        },
        "roles": len(all_jobs),
        "web_search_calls": context.search_calls,
        "page_fetch_calls": context.fetch_calls,
        "search_trace": {
            query: [
                {"title": item.title, "url": item.url, "snippet": item.snippet}
                for item in results[:5]
            ]
            for query, results in context.search_cache.items()
        },
        "source_warnings": source_warnings,
        "extraction_errors": extraction_errors,
        "jobs": [job.model_dump(mode="json") for job in all_jobs],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 104)
    print("LIVE CHECK SUMMARY")
    print("=" * 104)
    print("Roles        :", len(all_jobs))
    print("Web searches :", context.search_calls)
    print("Page fetches :", context.fetch_calls)
    print(
        "Full JDs     :",
        sum(
            job.jd_status in {"fetched_official", "fetched_secondary"}
            for job in all_jobs
        ),
    )
    print("Output       :", output)


if __name__ == "__main__":
    main()
