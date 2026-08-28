from career_agent.storage.sqlite import OpportunityStore


def test_store_persists_structured_record_and_deduplicates(tmp_path) -> None:
    store = OpportunityStore(tmp_path / "simplynext.db")
    job = {
        "company": "McKinsey & Company",
        "title": "Innovation and Learning Centre (ILC) Intern",
        "location": "Singapore",
        "opportunity_type": "internship",
        "official_url": None,
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
    assert "raw_description" not in saved
    assert "attachment_text" not in saved
