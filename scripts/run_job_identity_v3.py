from __future__ import annotations

import argparse

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.job_identity.discover_candidates import discover_candidates
from career_agent.job_identity.extract_identity import extract_job_identities
from career_agent.job_identity.verify_same_job import verify_same_job
from career_agent.nodes.extract_signal import extract_signal
from career_agent.nodes.normalize_email import normalize_email


def _short(value: str | None, limit: int = 120) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext V3: verify whether public candidates are the same job."
    )
    parser.add_argument("--scan", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--subject", default=None)
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable the single bounded LLM tie-breaker for ambiguous candidates.",
    )
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    if args.subject:
        needle = args.subject.lower()
        messages = [message for message in messages if needle in message.subject.lower()]
    messages = messages[: args.limit]

    print("=" * 100)
    print("SIMPLYNEXT JOB IDENTITY V3 — SAME-JOB VERIFICATION")
    print("=" * 100)
    print("Career emails selected:", len(messages))

    if not messages:
        raise RuntimeError("No matching career emails found.")

    for email_index, message in enumerate(messages, start=1):
        normalized = normalize_email(message)
        state = {
            "email": normalized.model_dump(mode="json"),
            "normalized_text": normalized.body_text,
            "extracted_links": normalized.links,
            "errors": [],
        }
        signal_result = extract_signal(state)
        signals = signal_result.get("opportunity_signals", [])
        identities = extract_job_identities(normalized, signals)

        print("\n" + "-" * 100)
        print(f"EMAIL {email_index}/{len(messages)}")
        print("Source :", f"{normalized.sender_name} <{normalized.sender_email}>")
        print("Subject:", _short(normalized.subject, 150))
        print("Signals:", len(signals))
        print("Identities:", len(identities.identities))

        for identity_index, identity in enumerate(identities.identities, start=1):
            discovery = discover_candidates(identity)
            result = verify_same_job(
                identity,
                discovery,
                normalized,
                enable_llm_judge=not args.no_llm_judge,
            )

            print("\n  " + "=" * 90)
            print(f"  IDENTITY {identity_index}/{len(identities.identities)}")
            print("  Company   :", identity.company)
            print("  Title     :", identity.title)
            print("  Location  :", identity.location)
            print("  IDs       :", ", ".join(i.value for i in identity.identifiers) or "none")

            print("\n  V2 SEARCH")
            for step in discovery.trace:
                print(
                    f"    Round {step.round_number}: {step.strategy} | "
                    f"{step.results_returned} results | {step.elapsed_ms} ms"
                )
                print("      Query:", step.query)
            print("    Candidates:", len(discovery.candidates))

            print("\n  V3 CANDIDATE EVIDENCE")
            if result.evaluations:
                for index, evaluation in enumerate(result.evaluations, start=1):
                    print(
                        f"    #{index} {evaluation.decision.upper()} "
                        f"score={evaluation.evidence_score:.1f} "
                        f"confidence={evaluation.confidence}"
                    )
                    print("      url      :", evaluation.final_url or evaluation.requested_url)
                    print("      title    :", _short(evaluation.page_title, 120))
                    print("      company  :", evaluation.company_match)
                    print("      title ov.:", f"{evaluation.title_overlap:.2f}")
                    print("      location :", evaluation.location_match)
                    print("      unit     :", evaluation.business_unit_match)
                    print("      IDs      :", ", ".join(evaluation.identifier_hits) or "none")
                    print(
                        "      phrases  :",
                        "; ".join(evaluation.distinctive_phrase_hits) or "none",
                    )
                    if evaluation.hard_conflicts:
                        print("      CONFLICTS:", "; ".join(evaluation.hard_conflicts))
                    if evaluation.fetch_error:
                        print("      fetch err:", evaluation.fetch_error)
            else:
                print("    No public page was readable enough to evaluate.")

            print("\n  FINAL SAME-JOB DECISION")
            print("    status     :", result.identity_status)
            print("    basis      :", result.identity_basis)
            print("    confidence :", result.confidence)
            print("    official   :", result.official_url)
            print("    apply      :", result.application_url)

            print("\n  COST / LATENCY")
            print("    V1 identity LLM calls:", identities.metrics.llm_calls)
            print("    V2 search calls      :", discovery.metrics.search_calls)
            print("    V2 LLM calls         :", discovery.metrics.llm_calls)
            print("    V3 fetch calls       :", result.metrics.fetch_calls)
            print("    V3 pages fetched     :", result.metrics.pages_fetched)
            print("    V3 judge LLM calls   :", result.metrics.llm_calls)
            print("    V3 latency           :", f"{result.metrics.elapsed_ms} ms")

            if result.errors:
                print("\n  ERRORS")
                for error in result.errors:
                    print("   -", error)


if __name__ == "__main__":
    main()
