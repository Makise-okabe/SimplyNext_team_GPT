from career_agent.config import Settings


def filter_email(state: dict) -> dict:
    email = state.get("email") or {}
    sender = (email.get("sender_email") or "").lower()
    subject = (email.get("subject") or "").lower()

    settings = Settings()

    trusted_sender = sender in settings.trusted_senders
    career_subject = any(
        keyword in subject
        for keyword in ("career", "job", "intern", "talentconnect", "enews", "industry opportunities")
    )

    return {"is_career_email": trusted_sender or career_subject}
