from __future__ import annotations

import json
from pathlib import Path

from career_agent.nodes.extract_signal import extract_signal
from career_agent.nodes.normalize_email import normalize_email
from scripts.test_real_email_normalizer import graph_item_to_email, load_graph_messages

RAW_EMAIL_PATH = Path("data/raw_email.json")
TARGET_SENDER = "zeli.goh@nus.edu.sg"
SUBJECT_HINTS = (
    "industry opportunities",
    "career opportunities",
    "highlighted engineering career opportunities",
)


def choose_message(raw_messages: list[dict]) -> dict:
    ranked: list[tuple[int, dict]] = []

    for raw in raw_messages:
        message = graph_item_to_email(raw)
        if message.sender_email != TARGET_SENDER:
            continue

        subject = message.subject.lower()
        score = sum(1 for hint in SUBJECT_HINTS if hint in subject)
        if score:
            ranked.append((score, raw))

    if not ranked:
        raise RuntimeError(
            "Could not find a Goh Ze Li career-opportunities email in raw_email.json."
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def main() -> None:
    if not RAW_EMAIL_PATH.exists():
        raise FileNotFoundError(f"Could not find {RAW_EMAIL_PATH}")

    raw_messages = load_graph_messages(RAW_EMAIL_PATH)
    raw = choose_message(raw_messages)
    message = normalize_email(graph_item_to_email(raw))

    print("Selected real email")
    print(f"  Sender: {message.sender_name} <{message.sender_email}>")
    print(f"  Subject: {message.subject}")
    print(f"  Links after normalization: {len(message.links)}")
    print()

    state = {
        "email": message.model_dump(mode="json"),
        "normalized_text": message.body_text,
        "extracted_links": message.links,
        "errors": [],
    }

    result = extract_signal(state)
    errors = result.get("errors", [])
    signals = result.get("opportunity_signals", [])

    if errors:
        print("ERRORS")
        for error in errors:
            print(f"  - {error}")
        print()

    print(f"Extracted {len(signals)} opportunity signals.\n")

    for index, signal in enumerate(signals, start=1):
        print("=" * 88)
        print(f"SIGNAL {index}")
        print(json.dumps(signal, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
