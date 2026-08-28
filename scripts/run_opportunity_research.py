from __future__ import annotations

import argparse

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.job_identity.extract_identity import extract_job_identities
from career_agent.job_identity.official_research import (
    focus_email_for_target,
    research_opportunity,
)
from career_agent.nodes.extract_signal import extract_signal
from career_agent.nodes.normalize_email import normalize_email


def _matches(value: str | None, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower() in (value or "").lower()


def _short(value: str | None, limit: int = 130) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext V5: official-first opportunity provenance and research."
    )
    parser.add_argument("--scan", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--company", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    if args.subject:
        needle = args.subject.lower()
        messages = [message for message in messages if needle in message.subject.lower()]
    messages = messages[: args.limit]

    print("=" * 108)
    print("SIMPLYNEXT V5 — OFFICIAL-FIRST OPPORTUNITY RESEARCH + PROVENANCE")
    print("=" * 108)
    print("Career emails selected:", len(messages))

    if not messages:
        raise RuntimeError("No matching career emails found.")

    for email_index, message in enumerate(messages, start=1):
        normalized = normalize_email(message)
        focused = focus_email_for_target(
            normalized,
            company=args.company,
            title=args.title,
        )
        state = {
            "email": focused.model_dump(mode="json"),
            "normalized_text": focused.body_text,
            "extracted_links": focused.links,
            "errors": [],
        }
        signal_result = extract_signal(state)
        signals = signal_result.get("opportunity_signals", [])
        signals = [
            signal
            for signal in signals
            if _matches(signal.get("company"), args.company)
            and _matches(signal.get("role_title"), args.title)
        ]

        identities = extract_job_identities(focused, signals)

        print("\n" + "-" * 108)
        print(f"EMAIL {email_index}/{len(messages)}")
        print("Source :", f"{normalized.sender_name} <{normalized.sender_email}>")
        print("Subject:", _short(normalized.subject, 155))
        print("Target company:", args.company or "<all>")
        print("Target title  :", args.title or "<all>")
        print("Signals       :", len(signals))
        print("Identities    :", len(identities.identities))

        for index, identity in enumerate(identities.identities, start=1):
            package = research_opportunity(identity, normalized)

            print("\n  " + "=" * 98)
            print(f"  OPPORTUNITY {index}/{len(identities.identities)}")
            print("  Company     :", identity.company)
            print("  Title       :", identity.title)
            print("  Type        :", identity.opportunity_type)
            print("  Record kind :", package.record_kind)

            print("\n  SEARCH TRACE — OFFICIAL FIRST")
            if package.trace:
                for step in package.trace:
                    print(
                        f"    Round {step.round_number} [{step.scope}] | "
                        f"{step.results_returned} results | "
                        f"official={step.official_results} | "
                        f"secondary={step.secondary_results} | "
                        f"{step.elapsed_ms} ms"
                    )
                    print("      Query:", step.query)
            else:
                print("    No search needed: direct source URL was sufficient.")

            print("\n  EVIDENCE")
            for candidate in package.candidates[:8]:
                print(
                    f"    [{candidate.tier.upper():13}] "
                    f"{candidate.relation:20} score={candidate.score:.1f}"
                )
                print("      host :", candidate.host)
                print("      title:", _short(candidate.title, 120))
                print("      url  :", candidate.url)

            print("\n  FINAL PACKAGE")
            print("    status      :", package.status)
            print("    basis       :", package.basis)
            print("    confidence  :", package.confidence)
            print("    official job:", package.official_job_url)
            print(
                "    official bg :",
                "; ".join(package.official_background_urls) or "None",
            )
            print(
                "    secondary   :",
                "; ".join(package.secondary_evidence_urls) or "None",
            )
            print("    apply       :", package.application_url)

            print("\n  ORIGINAL SOURCE")
            print("    sender      :", package.provenance.sender_email)
            print("    subject     :", _short(package.provenance.subject, 150))
            print("    Outlook link:", package.provenance.original_email_url)
            print(
                "    attachments :",
                "; ".join(package.provenance.attachment_names) or "None",
            )

            print("\n  METRICS")
            print("    web searches:", package.metrics.search_calls)
            print("    page fetches :", package.metrics.fetch_calls)
            print("    judge LLM    :", package.metrics.judge_llm_calls)
            print("    latency      :", f"{package.metrics.elapsed_ms} ms")

            if package.warnings:
                print("\n  WARNINGS (NON-FATAL)")
                for warning in package.warnings:
                    print("   -", warning)
            print("\n  Pipeline errors:", len(package.errors))
            for error in package.errors:
                print("   -", error)


if __name__ == "__main__":
    main()
