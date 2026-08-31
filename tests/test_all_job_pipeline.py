from __future__ import annotations

from career_agent import all_job_extraction
from career_agent import batch_sources
from career_agent.models.email import EmailMessage
from career_agent.models.signal import ExtractedOpportunity, ExtractedOpportunityBatch


def test_rich_html_source_preserves_anchor_url_without_duplicate_plain_body(monkeypatch) -> None:
    email = EmailMessage(
        message_id="m1",
        sender_email="talentconnect@se.nus.edu.sg",
        subject="eNews",
        body_text="short recovered payload",
        body_html=(
            "<html><body><p>short recovered payload</p>"
            "<p>Point72 Academy Investment Analyst Program "
            "<a href='https://careers.point72.com/CSJobDetail?jobCode=CPA-0014976'>Apply</a>"
            "</p></body></html>"
        ),
        links=["https://outlook.live.com/owa/?ItemID=x"],
    )

    corpus, links, documents, warnings = batch_sources.build_source_corpus(
        email,
        fetch_linked_pdfs=False,
    )

    assert "Point72 Academy Investment Analyst Program" in corpus
    assert "https://careers.point72.com/CSJobDetail?jobCode=CPA-0014976" in corpus
    assert corpus.count("SOURCE: EMAIL") == 1
    assert "https://careers.point72.com/CSJobDetail?jobCode=CPA-0014976" in links
    assert len(documents) == 1
    assert warnings == []


def test_linked_pdf_is_added_as_source_document(monkeypatch) -> None:
    pdf_url = "https://nus.edu.sg/cfg/docs/enews.pdf"
    monkeypatch.setattr(batch_sources, "_fetch_linked_pdf", lambda url: "Mastercard Graduate Analyst")

    email = EmailMessage(
        message_id="m2",
        sender_email="talentconnect@se.nus.edu.sg",
        subject="eNews",
        body_html=f"<a href='{pdf_url}'>Download eNews PDF</a>",
    )

    corpus, _, documents, warnings = batch_sources.build_source_corpus(email)

    assert "Mastercard Graduate Analyst" in corpus
    assert any(doc.source_type == "linked_pdf" and doc.url == pdf_url for doc in documents)
    assert warnings == []


def test_all_job_extraction_splits_multiple_roles_and_deduplicates_overlap(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_invoke(chunk: str) -> ExtractedOpportunityBatch:
        calls["count"] += 1
        return ExtractedOpportunityBatch(
            opportunities=[
                ExtractedOpportunity(
                    company="Goldilock",
                    role_title="Embedded Software Engineer",
                    opportunity_type="full_time",
                    urls=["https://goldilock.example/jobs/embedded"],
                    evidence_text="Goldilock - Embedded Software Engineer",
                ),
                ExtractedOpportunity(
                    company="Goldilock",
                    role_title="Electronics Engineer",
                    opportunity_type="full_time",
                    urls=["https://goldilock.example/jobs/electronics"],
                    evidence_text="Goldilock - Electronics Engineer",
                ),
            ]
        )

    monkeypatch.setattr(all_job_extraction, "_invoke", fake_invoke)

    # Deliberately force overlapping chunks; the same two roles are returned from
    # every chunk and must collapse to two final opportunities.
    corpus = "Goldilock careers " * 900
    opportunities, metrics, errors = all_job_extraction.extract_all_opportunities(
        source_name="Goh Ze Li",
        source_message_id="m3",
        source_date=None,
        corpus=corpus,
    )

    assert len(opportunities) == 2
    assert {item.role_title for item in opportunities} == {
        "Embedded Software Engineer",
        "Electronics Engineer",
    }
    assert metrics.llm_calls == calls["count"]
    assert metrics.source_chars == len(corpus)
    assert errors == []
