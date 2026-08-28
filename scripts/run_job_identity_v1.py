from __future__ import annotations

import argparse

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.job_identity.extract_identity import extract_job_identities
from career_agent.nodes.extract_signal import extract_signal
from career_agent.nodes.normalize_email import normalize_email


def _short(value: str | None, limit: int = 110) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext V1: extract compact job identities from live career email."
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

    print("=" * 92)
    print("SIMPLYNEXT JOB IDENTITY V1")
    print("=" * 92)
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
        signal_errors = signal_result.get("errors", [])

        print("\n" + "-" * 92)
        print(f"EMAIL {email_index}/{len(messages)}")
        print("Source :", f"{normalized.sender_name} <{normalized.sender_email}>")
        print("Subject:", _short(normalized.subject, 140))
        print("Signals:", len(signals))

        if signal_errors:
            print("Signal extraction errors:")
            for error in signal_errors:
                print(" -", error)

        result = extract_job_identities(normalized, signals)

        for identity_index, identity in enumerate(result.identities, start=1):
            print("\n  " + "=" * 82)
            print(f"  JOB IDENTITY {identity_index}/{len(result.identities)}")
            print("  Company       :", identity.company)
            print("  Title         :", identity.title)
            print("  Location      :", identity.location)
            print("  Type          :", identity.opportunity_type)
            print("  Strength      :", identity.identity_strength)
            print("  Business unit :", identity.business_unit)
            print("  Team          :", identity.team)
            print("  Duration      :", identity.duration)
            print("  Start period  :", identity.start_period)
            print("  End period    :", identity.end_period)
            print("  Target cohort :", "; ".join(identity.target_cohort) or "none")

            if identity.identifiers:
                print("  Identifiers   :")
                for identifier in identity.identifiers:
                    print(
                        f"    - {identifier.kind}: "
                        f"{identifier.label} = {identifier.value}"
                    )
            else:
                print("  Identifiers   : none")

            print("  Direct URLs   :", len(identity.direct_urls))
            for url in identity.direct_urls[:5]:
                print("    -", url)

            print("  Distinctive phrases:")
            if identity.distinctive_phrases:
                for phrase in identity.distinctive_phrases:
                    print("    -", phrase)
            else:
                print("    - none extracted")

            print("  Fingerprint   :", identity.source_fingerprint[:16] + "...")

        metrics = result.metrics
        print("\n  V1 METRICS")
        print("  Signals seen      :", metrics.signals_seen)
        print("  Identities built  :", metrics.identities_built)
        print("  Identity LLM calls:", metrics.llm_calls)
        print("  Batches           :", metrics.batches)
        print("  Source chars sent :", metrics.source_chars_sent)
        print("  Identity latency  :", f"{metrics.elapsed_ms} ms")
        print("  Web search calls  : 0  (V1 deliberately does not search the web)")

        if result.errors:
            print("  Identity extraction errors:")
            for error in result.errors:
                print("   -", error)


if __name__ == "__main__":
    main()
