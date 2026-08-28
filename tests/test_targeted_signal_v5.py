from datetime import datetime, timezone

from career_agent.job_identity.targeted_signal import build_targeted_signal
from career_agent.models.email import EmailMessage


def test_targeted_signal_recovers_exact_ibm_row_and_job_url() -> None:
    html = """
    <table>
      <tr>
        <td>Information Communication Technology</td>
        <td>IBM</td>
        <td>Associate Application Developer-AWS Cloud (Based in Bangkok)</td>
        <td><a href="https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733&source=WEB_Search_NA">Apply here</a></td>
      </tr>
      <tr>
        <td></td><td>IBM</td>
        <td>Associate Application Consultant-Business Functions (Based in Bangkok)</td>
        <td><a href="https://careers.ibm.com/en_US/careers/JobDetail?jobId=88762&source=WEB_Search_NA">Apply here</a></td>
      </tr>
    </table>
    """
    email = EmailMessage(
        message_id="m1",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        received_at=datetime(2026, 2, 8, tzinfo=timezone.utc),
        body_text=(
            "IBM Associate Application Developer-AWS Cloud (Based in Bangkok) "
            "IBM Associate Application Consultant-Business Functions (Based in Bangkok)"
        ),
        body_html=html,
        links=[
            "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733&source=WEB_Search_NA",
            "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88762&source=WEB_Search_NA",
        ],
    )

    signal = build_targeted_signal(
        email,
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
    )

    assert signal is not None
    assert signal.company == "IBM"
    assert signal.role_title == "Associate Application Developer-AWS Cloud"
    assert signal.location == "Bangkok"
    assert signal.opportunity_type == "full_time"
    assert signal.urls == [
        "https://careers.ibm.com/en_US/careers/JobDetail?jobId=88733&source=WEB_Search_NA"
    ]
    assert "88762" not in " ".join(signal.urls)


def test_targeted_signal_requires_exact_title_presence() -> None:
    email = EmailMessage(
        message_id="m1",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        body_text="Marvell Senior Staff Analog Layout Engineer",
    )

    assert build_targeted_signal(
        email,
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
    ) is None
