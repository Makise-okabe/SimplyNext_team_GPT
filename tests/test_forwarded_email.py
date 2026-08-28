from career_agent.models.email import EmailMessage
from career_agent.parsers.forwarded_email import recover_forwarded_email


def test_recovers_goh_forward_and_preserves_forwarder() -> None:
    message = EmailMessage(
        message_id="m1",
        sender_name="Du Yanzhang",
        sender_email="student@u.nus.edu",
        subject="Fw: [By 30 March] 6-month Internship Opportunity",
        body_text=(
            "Get Outlook for iOS\n"
            "________________________________\n"
            "From: Goh Ze Li <zeli.goh@nus.edu.sg>\n"
            "Sent: Friday, 27 March 2026 17:24:29\n"
            "Cc: Kirby Ken Mark <kirbym@nus.edu.sg>\n"
            "Subject: [By 30 March] 6-month Internship Opportunity\n\n"
            "Dear Students,\nPlease see attached JD."
        ),
    )

    recovered = recover_forwarded_email(message)

    assert recovered.sender_email == "zeli.goh@nus.edu.sg"
    assert recovered.sender_name == "Goh Ze Li"
    assert recovered.transport_sender_email == "student@u.nus.edu"
    assert recovered.subject == "[By 30 March] 6-month Internship Opportunity"
    assert "Dear Students" in recovered.body_text


def test_recovers_talentconnect_forward() -> None:
    message = EmailMessage(
        message_id="m2",
        sender_name="Fang Tianchi",
        sender_email="fang.tianchi@u.nus.edu",
        subject="Fwd: eNews: SMRT, VIAVI Solutions and more!",
        body_text=(
            "From: Nus talentconnect <talentconnect@se.nus.edu.sg>\n"
            "Sent: Wednesday, 19 August 2026 15:39:05\n"
            "To: Fang Tianchi <fang.tianchi@u.nus.edu>\n"
            "Subject: eNews: SMRT, VIAVI Solutions and more!\n\n"
            "Career opportunities here."
        ),
    )

    recovered = recover_forwarded_email(message)

    assert recovered.sender_email == "talentconnect@se.nus.edu.sg"
    assert recovered.transport_sender_email == "fang.tianchi@u.nus.edu"
    assert recovered.subject == "eNews: SMRT, VIAVI Solutions and more!"
