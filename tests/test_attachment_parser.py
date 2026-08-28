from career_agent.parsers.attachments import normalize_attachment_text


def test_normalize_attachment_text_joins_hyphenated_layout_breaks() -> None:
    text = "Advanced manu-\nfacturing\n\n\nAI   and   IoT"
    normalized = normalize_attachment_text(text)

    assert "manufacturing" in normalized
    assert "\n\n\n" not in normalized
    assert "AI and IoT" in normalized
