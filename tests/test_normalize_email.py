from career_agent.models.email import EmailMessage
from career_agent.nodes.normalize_email import (
    extract_links_from_html,
    html_to_text,
    normalize_email,
)


def test_html_to_text_removes_markup_and_script() -> None:
    html = """
    <html>
      <body>
        <h1>AMD Graduate Role</h1>
        <p>Apply before 30 September.</p>
        <script>ignore_me()</script>
      </body>
    </html>
    """

    text = html_to_text(html)

    assert "AMD Graduate Role" in text
    assert "Apply before 30 September." in text
    assert "ignore_me" not in text


def test_extract_links_from_html_deduplicates_and_keeps_order() -> None:
    html = """
    <a href="https://example.com/job">Job</a>
    <a href="https://example.com/job">Duplicate</a>
    <a href="https://example.com/apply">Apply</a>
    """

    assert extract_links_from_html(html) == [
        "https://example.com/job",
        "https://example.com/apply",
    ]


def test_normalize_email_builds_text_and_merges_links() -> None:
    message = EmailMessage(
        message_id="msg-1",
        sender_email="no-reply@kinobi.asia",
        subject="Career opportunity",
        body_html=(
            '<p>Micron internship</p>'
            '<a href="https://careers.example/job-1">View role</a>'
        ),
        links=["https://outlook.office.com/mail/deeplink/msg-1"],
    )

    normalized = normalize_email(message)

    assert normalized.body_text == "Micron internship\nView role"
    assert normalized.links == [
        "https://outlook.office.com/mail/deeplink/msg-1",
        "https://careers.example/job-1",
    ]
