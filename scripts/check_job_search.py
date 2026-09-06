from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.job_link_resolver import resolve_job_link
from career_agent.models.job_record import JobRecord
from career_agent.research_session import research_session


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
    with research_session() as session:
        resolved, result = resolve_job_link(job)

    print("SIMPLYNEXT COMPLETE JOB SEARCH")
    print(f"Company    : {args.company}")
    print(f"Role       : {args.title}")
    print(f"Searches   : {session.search_calls}")
    print(f"Pages read : {session.fetch_calls}")
    for attempt in resolved.link_attempts:
        print(f"  {attempt.get('status', 'unknown'):12} | {attempt.get('final_url') or attempt.get('url')}")
    if not result.url:
        raise SystemExit("No verified exact role page found across the complete search and site-navigation flow.")
    print(f"Result     : {result.kind} / {result.confidence}")
    print(f"URL        : {result.url}")


if __name__ == "__main__":
    main()
