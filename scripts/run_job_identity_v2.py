from __future__ import annotations

import argparse

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.job_identity.discover_candidates import discover_candidates
from career_agent.job_identity.extract_identity import extract_job_identities
from career_agent.nodes.extract_signal import extract_signal
from career_agent.nodes.normalize_email import normalize_email


def _short(value: str | None, limit: int = 120) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext V2: progressive candidate discovery for exact-role verification."
    )
    parser.add_argument("--scan", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--subject", default=None)
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)

    if args.subject:
        needle = args.subject.lower()
        messages = [message for message in messages if needle in message.subject.lower()]
    messages = messages[: args.limit]

    print("=" * 96)
    print("SIMPLYNEXT JOB IDENTITY V2 — PROGRESSIVE CANDIDATE DISCOVERY")
    print("=" * 96)
    print("Career emails selected:", len(messages))

    if not messages:
        raise RuntimeError("No matching career emails found.")

    for email_index, message in enumerate(messages, start=1):
        normalized = normalize_email(message)
        signal_state = {
            "email": normalized.model_dump(mode="json"),
            "normalized_text": normalized.body_text,
            "extracted_links": normalized.links,
            "errors": [],
        }
        signal_result = extract_signal(signal_state)
        signals = signal_result.get("opportunity_signals", [])
        identity_result = extract_job_identities(normalized, signals)

        print("\n" + "-" * 96)
        print(f"EMAIL {email_index}/{len(messages)}")
        print("Source :", f"{normalized.sender_name} <{normalized.sender_email}>")
        print("Subject:", _short(normalized.subject, 150))
        print("Signals:", len(signals))
        print("Identities:", len(identity_result.identities))

        for identity_index, identity in enumerate(identity_result.identities, start=1):
            print("\n  " + "=" * 86)
            print(f"  IDENTITY {identity_index}/{len(identity_result.identities)}")
            print("  Company   :", identity.company)
            print("  Title     :", identity.title)
            print("  Location  :", identity.location)
            print("  Strength  :", identity.identity_strength)
            print("  IDs       :", ", ".join(i.value for i in identity.identifiers) or "none")
            print("  Direct URLs:", len(identity.direct_urls))

            discovery = discover_candidates(identity)

            print("\n  SEARCH TRACE")
            if discovery.trace:
                for step in discovery.trace:
                    print(
                        f"    Round {step.round_number}: {step.strategy} | "
                        f"{step.elapsed_ms} ms | {step.results_returned} results"
                    )
                    print("      Query:", step.query)
            else:
                print("    No web search needed before candidate handoff.")

            print("\n  CANDIDATES FOR V3")
            if not discovery.candidates:
                print("    none")
            for rank, candidate in enumerate(discovery.candidates, start=1):
                print(f"    #{rank} score={candidate.discovery_score:.2f}")
                print("      title :", _short(candidate.title, 130))
                print("      host  :", candidate.host)
                print("      kind  :", candidate.url_kind)
                print("      url   :", candidate.url)
                print("      via   :", ", ".join(candidate.strategies))
                if candidate.identifier_hits:
                    print("      ID hit:", ", ".join(candidate.identifier_hits))
                if candidate.distinctive_phrase_hits:
                    print(
                        "      phrase:",
                        "; ".join(candidate.distinctive_phrase_hits[:3]),
                    )
                if candidate.metadata_hits:
                    print("      meta  :", ", ".join(candidate.metadata_hits))

            metrics = discovery.metrics
            print("\n  V2 METRICS")
            print("    Web search calls :", metrics.search_calls)
            print("    Raw results      :", metrics.raw_results_seen)
            print("    Unique candidates:", metrics.unique_candidates)
            print("    Final candidates :", len(discovery.candidates))
            print("    V2 LLM calls     :", metrics.llm_calls)
            print("    Search latency   :", f"{metrics.elapsed_ms} ms")
            print("    Stop reason      :", metrics.stopped_reason)
            print("    Verification     : NOT DECIDED (V3 responsibility)")

            if discovery.errors:
                print("    Errors:")
                for error in discovery.errors:
                    print("      -", error)


if __name__ == "__main__":
    main()
