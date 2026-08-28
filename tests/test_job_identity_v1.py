from career_agent.job_identity.extract_identity import (
    IdentityInput,
    _build_identity,
    _chunk_inputs,
    build_signal_context,
    extract_identifiers,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import ExtractedJobIdentity


def test_extracts_common_job_identifier_labels() -> None:
    text = (
        "Job ID: JR123456\n"
        "Requisition ID: R-98765\n"
        "Posting No. P_7788\n"
        "Reference Number: REF-22A"
    )

    identifiers = extract_identifiers(text)
    values = {(item.kind, item.value) for item in identifiers}

    assert ("job_id", "JR123456") in values
    assert ("requisition_id", "R-98765") in values
    assert ("posting_id", "P_7788") in values
    assert ("reference_number", "REF-22A") in values


def test_context_is_role_specific_and_bounded() -> None:
    noise = "unrelated newsletter text " * 500
    role = (
        "McKinsey Innovation and Learning Centre Intern in Singapore. "
        "The intern will use a Supply Chain Control Tower and digital war room."
    )
    message = EmailMessage(
        message_id="m1",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Internship opportunity",
        body_text=f"{noise}\n{role}\n{noise}",
    )
    signal = {
        "company": "McKinsey",
        "role_title": "Innovation and Learning Centre Intern",
        "location": "Singapore",
        "opportunity_type": "internship",
        "raw_text": "Supply Chain Control Tower",
        "urls": [],
    }

    context = build_signal_context(message, signal)

    assert "Supply Chain Control Tower" in context
    assert "Innovation and Learning Centre Intern" in context
    assert len(context) <= 4200
    assert len(context) < len(message.body_text)


def test_grounding_drops_hallucinated_distinctive_phrase() -> None:
    context = (
        "McKinsey & Company Innovation and Learning Centre Intern Singapore. "
        "The role includes a Supply Chain Control Tower and digital war room."
    )
    message = EmailMessage(
        message_id="m2",
        sender_email="zeli.goh@nus.edu.sg",
        subject="ILC Internship",
        body_text=context,
    )
    signal = {
        "company": "McKinsey & Company",
        "role_title": "Innovation and Learning Centre Intern",
        "location": "Singapore",
        "opportunity_type": "internship",
        "raw_text": "Supply Chain Control Tower",
        "urls": ["https://forms.office.com/r/example"],
    }
    extracted = ExtractedJobIdentity(
        source_index=1,
        company="McKinsey & Company",
        title="Innovation and Learning Centre Intern",
        location="Singapore",
        distinctive_phrases=[
            "Supply Chain Control Tower",
            "digital war room",
            "quantum blockchain laboratory",
        ],
    )
    input_item = IdentityInput(source_index=1, signal=signal, context=context)

    identity = _build_identity(message, input_item, extracted)

    assert "Supply Chain Control Tower" in identity.distinctive_phrases
    assert "digital war room" in identity.distinctive_phrases
    assert "quantum blockchain laboratory" not in identity.distinctive_phrases
    assert identity.identity_strength == "strong"


def test_identifier_makes_identity_strong_and_fingerprint_stable() -> None:
    context = "AMD Product Development Engineer Singapore. Job Req ID: JR-778899"
    message = EmailMessage(
        message_id="m3",
        sender_email="zeli.goh@nus.edu.sg",
        subject="AMD role",
        body_text=context,
    )
    signal = {
        "company": "AMD",
        "role_title": "Product Development Engineer",
        "location": "Singapore",
        "opportunity_type": "full_time",
        "raw_text": context,
        "urls": [],
    }
    item = IdentityInput(source_index=1, signal=signal, context=context)

    first = _build_identity(message, item, None)
    second = _build_identity(message, item, None)

    assert first.identifiers[0].value == "JR-778899"
    assert first.identity_strength == "strong"
    assert first.source_fingerprint == second.source_fingerprint


def test_batching_caps_signals_and_total_context() -> None:
    inputs = [
        IdentityInput(
            source_index=index,
            signal={"company": f"Company {index}"},
            context="x" * 4000,
        )
        for index in range(1, 7)
    ]

    batches = _chunk_inputs(inputs)

    assert len(batches) == 2
    assert all(len(batch) <= 4 for batch in batches)
    assert all(sum(len(item.context) for item in batch) <= 15000 for batch in batches)


def test_llm_schema_accepts_null_optional_fields() -> None:
    extracted = ExtractedJobIdentity(
        source_index=1,
        company=None,
        title=None,
        location=None,
        business_unit=None,
        team=None,
        duration=None,
        distinctive_phrases=None,
    )

    assert extracted.source_index == 1
    assert extracted.distinctive_phrases is None
