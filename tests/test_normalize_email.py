from career_agent.models.email import EmailMessage
from career_agent.nodes.normalize_email import (
    extract_links_from_html,
    extract_links_from_text,
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


def test_extract_links_from_text_finds_urls_and_strips_punctuation() -> None:
    text = (
        "See https://www.tata.com/careers/programs/tata-global-internships. "
        "Apply at https://example.com/job?id=123). "
        "Duplicate: https://www.tata.com/careers/programs/tata-global-internships"
    )

    assert extract_links_from_text(text) == [
        "https://www.tata.com/careers/programs/tata-global-internships",
        "https://example.com/job?id=123",
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


def test_normalize_plain_text_email_extracts_inline_urls() -> None:
    message = EmailMessage(
        message_id="msg-2",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        body_text=(
            "Tata Global Internship: "
            "https://www.tata.com/careers/programs/tata-global-internships\n"
            "CDE internship info: "
            "https://cde.nus.edu.sg/undergraduate/engineering-internships/student-info/"
        ),
        links=["https://outlook.office.com/mail/deeplink/msg-2"],
    )

    normalized = normalize_email(message)

    assert normalized.links == [
        "https://outlook.office.com/mail/deeplink/msg-2",
        "https://www.tata.com/careers/programs/tata-global-internships",
        "https://cde.nus.edu.sg/undergraduate/engineering-internships/student-info/",
    ]
