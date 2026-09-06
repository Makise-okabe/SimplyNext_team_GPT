from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.job_link_resolver import _score_result
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search_aggregate import search_public_web_aggregated


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test SimplyNext's complete mixed-provider job search.")
    parser.add_argument("--company", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    job = JobRecord(
        source_key="unknown",
        source_message_id="search-check",
        source_subject="Search quality check",
        company=args.company,
        title=args.title,
        availability_status="active_candidate",
        record_kind="job_posting",
    )
    query = f'"{args.company}" "{args.title}" careers job Singapore'
    rows = search_public_web_aggregated(
        query,
        max_results=16,
        min_results=8,
        strict_relevance=False,
    )
    ranked = []
    for row in rows:
        score = _score_result(job, row)
        if score:
            ranked.append((score[0], score[1], row))
    ranked.sort(key=lambda item: item[0], reverse=True)

    print("SIMPLYNEXT COMPLETE JOB SEARCH")
    print(f"Query      : {query}")
    print(f"Candidates : {len(rows)} raw / {len(ranked)} company-title matches")
    if not ranked:
        raise SystemExit("No usable company-title match found across all configured providers.")
    for index, (score, kind, row) in enumerate(ranked[:8], start=1):
        print(f"{index:02d}. {kind} score={score:.1f} | {row.title}")
        print(f"    {row.url}")


if __name__ == "__main__":
    main()
