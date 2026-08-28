from __future__ import annotations

import argparse
from pathlib import Path

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.job_identity.v4_pipeline import process_email_v4
from career_agent.nodes.normalize_email import normalize_email
from career_agent.storage.verification_cache import VerificationCacheStore


def _short(value: str | None, limit: int = 130) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext V4: cached same-job verification + liveness + metrics."
    )
    parser.add_argument("--scan", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--subject", default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("private_data/job_identity_v4.db"),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat processing in one run to demonstrate warm-cache behaviour.",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Clear only the V4 structured cache before running.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass V4 source/verification/liveness cache for this run.",
    )
    args = parser.parse_args()

    cache = VerificationCacheStore(args.cache)
    if args.reset_cache:
        cache.clear_all()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    if args.subject:
        needle = args.subject.lower()
        messages = [message for message in messages if needle in message.subject.lower()]
    messages = messages[: args.limit]

    print("=" * 104)
    print("SIMPLYNEXT JOB IDENTITY V4 — CACHE + LIVENESS + COST/LATENCY")
    print("=" * 104)
    print("Career emails selected:", len(messages))
    print("Cache:", args.cache)
    if args.reset_cache:
        print("Cache reset: YES")

    if not messages:
        raise RuntimeError("No matching career emails found.")

    for email_index, message in enumerate(messages, start=1):
        normalized = normalize_email(message)
        print("\n" + "-" * 104)
        print(f"EMAIL {email_index}/{len(messages)}")
        print("Source :", f"{normalized.sender_name} <{normalized.sender_email}>")
        print("Subject:", _short(normalized.subject, 155))

        for pass_number in range(1, max(1, args.repeat) + 1):
            result = process_email_v4(
                normalized,
                cache,
                force_refresh=args.force_refresh,
            )
            metrics = result.metrics

            print("\n  " + "=" * 94)
            print(f"  V4 PASS {pass_number}/{max(1, args.repeat)}")
            print("  Source identity cache:", "HIT" if metrics.source_identity_cache_hit else "MISS")

            for index, outcome in enumerate(result.outcomes, start=1):
                identity = outcome.identity
                verification = outcome.verification
                liveness = outcome.liveness
                print(f"\n    JOB {index}/{len(result.outcomes)}")
                print("      company   :", identity.company)
                print("      title     :", identity.title)
                print("      verify    :", verification.identity_status)
                print("      basis     :", verification.identity_basis)
                print("      confidence:", verification.confidence)
                print("      official  :", verification.official_url)
                print("      apply     :", verification.application_url)
                print(
                    "      V3 cache   :",
                    "HIT" if outcome.verification_cache_hit else "MISS",
                )
                print("      liveness  :", liveness.status)
                print("      live reason:", liveness.reason)
                if liveness.url:
                    print(
                        "      live cache :",
                        "HIT" if outcome.liveness_cache_hit else "MISS",
                    )

            print("\n  V4 COST / LATENCY")
            print("    verification cache hits :", metrics.verification_cache_hits)
            print("    verification cache miss :", metrics.verification_cache_misses)
            print("    liveness cache hits     :", metrics.liveness_cache_hits)
            print("    liveness cache miss     :", metrics.liveness_cache_misses)
            print("    signal LLM calls        :", metrics.signal_llm_calls)
            print("    identity LLM calls      :", metrics.identity_llm_calls)
            print("    V2 search calls         :", metrics.search_calls)
            print("    V3 candidate fetches    :", metrics.verification_fetch_calls)
            print("    V3 judge LLM calls      :", metrics.judge_llm_calls)
            print("    liveness fetches        :", metrics.liveness_fetch_calls)
            print("    TOTAL LLM calls         :", metrics.total_llm_calls)
            print(
                "    approx LLM source chars :",
                metrics.approx_llm_source_chars,
                "(not billed-token count)",
            )
            print("    V4 processing latency   :", f"{metrics.elapsed_ms} ms")

            if result.warnings:
                print("\n  WARNINGS (NON-FATAL)")
                for warning in result.warnings:
                    print("   -", warning)

            print("\n  Pipeline errors:", len(result.errors))
            for error in result.errors:
                print("   -", error)

    print("\nNo raw email body, PDF text, attachment bytes or fetched-page body was cached.")


if __name__ == "__main__":
    main()
