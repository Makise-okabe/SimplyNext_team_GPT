from career_agent.storage.sqlite import OpportunityStore


def test_store_persists_structured_record_and_deduplicates(tmp_path) -> None:
    store = OpportunityStore(tmp_path / "simplynext.db")
    job = {
        "company": "McKinsey & Company",
        "title": "Innovation and Learning Centre (ILC) Intern",
        "location": "Singapore",
        "opportunity_type": "internship",
        "official_url": None,
        "application_url": "https://forms.office.com/r/example",
        "deadline": "2026-03-30",
        "verification_status": "source_verified",
        "verification_basis": "trusted_email_attachment",
        "raw_description": "THIS RAW JD MUST NOT BE STORED",
    }
    source = {
        "sender_name": "Goh Ze Li",
        "sender_email": "zeli.goh@nus.edu.sg",
        "message_id": "message-1",
        "received_at": "2026-03-27T17:24:29+00:00",
        "attachment_text": "THIS ATTACHMENT TEXT MUST NOT BE STORED",
    }

    assert store.upsert_job(job, source) is True
    assert store.upsert_job(job, source) is False
    assert store.count() == 1

    saved = store.list_recent(limit=10)[0]
    assert saved["company"] == "McKinsey & Company"
    assert saved["verification_status"] == "source_verified"
    assert saved["application_url"] == "https://forms.office.com/r/example"
    assert "raw_description" not in saved
    assert "attachment_text" not in saved


def test_store_updates_same_job_when_official_url_is_found_later(tmp_path) -> None:
    store = OpportunityStore(tmp_path / "simplynext.db")
    source_job = {
        "company": "Example Corp",
        "title": "Graduate Engineer",
        "location": "Singapore",
        "opportunity_type": "full_time",
        "official_url": None,
        "verification_status": "source_verified",
        "verification_basis": "trusted_email_attachment",
    }
    web_job = {
        **source_job,
        "official_url": "https://careers.example.com/jobs/123",
        "verification_status": "verified",
        "verification_basis": "official_web",
    }

    assert store.upsert_job(source_job) is True
    assert store.upsert_job(web_job) is False
    assert store.count() == 1

    saved = store.list_recent()[0]
    assert saved["official_url"] == "https://careers.example.com/jobs/123"
    assert saved["verification_status"] == "verified"
    assert saved["verification_basis"] == "official_web"


def test_store_does_not_downgrade_verification(tmp_path) -> None:
    store = OpportunityStore(tmp_path / "simplynext.db")
    verified = {
        "company": "AMD",
        "title": "Product Development Engineer",
        "location": "Singapore",
        "opportunity_type": "full_time",
        "official_url": "https://careers.amd.com/job/123",
        "verification_status": "verified",
        "verification_basis": "official_web",
    }
    later_partial = {
        **verified,
        "official_url": None,
        "verification_status": "partial",
        "verification_basis": "public_web",
    }

    store.upsert_job(verified)
    store.upsert_job(later_partial)

    saved = store.list_recent()[0]
    assert saved["verification_status"] == "verified"
    assert saved["verification_basis"] == "official_web"
