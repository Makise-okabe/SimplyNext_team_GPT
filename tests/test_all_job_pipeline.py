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


def test_linked_pdf_pages_are_emitted_as_separate_source_documents(monkeypatch) -> None:
    pdf_url = "https://nus.edu.sg/cfg/docs/enews.pdf"
    job_url = "https://careers.example.com/job/12345"
    pages = [
        "[[SIMPLYNEXT_PDF_PAGE_START:1]]\nMastercard Graduate Analyst\n[[SIMPLYNEXT_PDF_PAGE_END]]",
        (
            "[[SIMPLYNEXT_PDF_PAGE_START:2]]\nPoint72 Academy Internship\n"
            "[[SIMPLYNEXT_PDF_PAGE_LINKS]]\n"
            f"<{job_url}>\n[[SIMPLYNEXT_PDF_PAGE_END]]"
        ),
    ]
    monkeypatch.setattr(
        batch_sources,
        "_fetch_linked_pdf",
        lambda url: (pages, [job_url]),
    )

    email = EmailMessage(
        message_id="m2",
        sender_email="talentconnect@se.nus.edu.sg",
        subject="eNews",
        body_html=f"<a href='{pdf_url}'>Download eNews PDF</a>",
    )

    corpus, links, documents, warnings = batch_sources.build_source_corpus(email)

    assert "Mastercard Graduate Analyst" in corpus
    assert "Point72 Academy Internship" in corpus
    assert job_url in corpus
    assert job_url in links
    assert corpus.count("SOURCE: LINKED PDF PAGE") == 2
    assert corpus.count(batch_sources.SOURCE_DOCUMENT_SEPARATOR) == 2
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


def test_failed_dense_chunk_is_adaptively_split_and_recovered(monkeypatch) -> None:
    calls: list[int] = []

    def fake_invoke(chunk: str) -> ExtractedOpportunityBatch:
        calls.append(len(chunk))
        if len(chunk) > 4000:
            raise RuntimeError("Failed to parse tool call arguments as JSON")
        title = "Associate, Singapore (2027)" if "LEFTROLE" in chunk else "Consultant, Singapore (2027)"
        return ExtractedOpportunityBatch(
            opportunities=[
                ExtractedOpportunity(
                    company="BCG",
                    role_title=title,
                    opportunity_type="full_time",
                    evidence_text=f"BCG - {title}",
                )
            ]
        )

    monkeypatch.setattr(all_job_extraction, "_invoke", fake_invoke)

    corpus = ("LEFTROLE BCG careers\n" * 170) + ("RIGHTROLE BCG careers\n" * 170)
    opportunities, metrics, errors = all_job_extraction.extract_all_opportunities(
        source_name="Goh Ze Li",
        source_message_id="m4",
        source_date=None,
        corpus=corpus,
    )

    assert {item.role_title for item in opportunities} == {
        "Associate, Singapore (2027)",
        "Consultant, Singapore (2027)",
    }
    assert metrics.llm_calls == len(calls)
    assert any(size > 4000 for size in calls)
    assert any(size <= 4000 for size in calls)
    assert errors == []


def test_structured_jobs_and_internships_tables_are_parsed_without_llm(monkeypatch) -> None:
    html = """
    <html><body>
      <h3>JOBS</h3>
      <table><tbody>
        <tr><th>INDUSTRY</th><th>COMPANY</th><th>ROLE</th><th>TC ID</th><th>REMARKS</th></tr>
        <tr><td>ICT</td><td>Amazon Singapore</td><td>Program Manager</td><td>275001</td><td>Location: Singapore</td></tr>
      </tbody></table>
      <h3>INTERNSHIPS</h3>
      <table><tbody>
        <tr><th>INDUSTRY</th><th>COMPANY</th><th>ROLE</th><th>TC ID</th><th>REMARKS</th></tr>
        <tr><td>ICT</td><td>Amazon Singapore</td><td>Program Manager Intern - AWS Cloud</td><td>275002</td><td>Deadline: 24 Dec 25</td></tr>
        <tr><td>ICT</td><td>BLACK SESAME TECHNOLOGIES (SINGAPORE) PTE LTD</td><td>AI Engineer</td><td>275003</td><td>Deadline: 28 Dec 25</td></tr>
        <tr><td>ICT</td><td>Google Asia Pacific Pte Ltd</td><td>Data Center Technician Intern, 2026</td><td>275004</td><td>Summer Vacation</td></tr>
        <tr><td>ICT</td><td>Mikomiko Pte Ltd</td><td>AI Engineer (Machine Learning)</td><td>275005</td><td>Winter Vacation</td></tr>
      </tbody></table>
    </body></html>
    """

    email = EmailMessage(
        message_id="m5",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        body_html=html,
    )
    corpus, _, _, warnings = batch_sources.build_source_corpus(email, fetch_linked_pdfs=False)

    def no_llm(_chunk: str):
        raise AssertionError("structured table rows should not require LLM extraction")

    monkeypatch.setattr(all_job_extraction, "_invoke", no_llm)
    opportunities, metrics, errors = all_job_extraction.extract_all_opportunities(
        source_name="Goh Ze Li",
        source_message_id="m5",
        source_date=None,
        corpus=corpus,
    )

    pairs = {(item.company, item.role_title): item for item in opportunities}
    assert ("Amazon Singapore", "Program Manager") in pairs
    assert pairs[("Amazon Singapore", "Program Manager")].opportunity_type == "full_time"
    assert ("Amazon Singapore", "Program Manager Intern - AWS Cloud") in pairs
    assert pairs[("Amazon Singapore", "Program Manager Intern - AWS Cloud")].opportunity_type == "internship"
    assert ("BLACK SESAME TECHNOLOGIES (SINGAPORE) PTE LTD", "AI Engineer") in pairs
    assert ("Google Asia Pacific Pte Ltd", "Data Center Technician Intern, 2026") in pairs
    assert ("Mikomiko Pte Ltd", "AI Engineer (Machine Learning)") in pairs
    assert metrics.llm_calls == 0
    assert warnings == []
    assert errors == []


def test_rowspan_continuations_keep_company_and_events_are_excluded(monkeypatch) -> None:
    html = """
    <html><body>
      <h3>JOBS</h3>
      <table><tbody>
        <tr><th>INDUSTRY</th><th>COMPANY</th><th>ROLE</th><th>TC ID</th><th>REMARKS</th></tr>
        <tr>
          <td>ICT</td>
          <td rowspan='3'>OPPO * for the roles in Qianhai, candidates will be based in Shenzhen</td>
          <td>Site Reliability Engineer</td><td>272792</td><td>Location: Singapore</td>
        </tr>
        <tr><td>[Qianhai] Intelligent Manufacturing Engineer (Process Direction)</td><td>275095</td><td>Location: Shenzhen, China</td></tr>
        <tr><td>[Qianhai] Software Product Manager (Overseas)</td><td>275085</td><td>Location: Shenzhen, China</td></tr>
      </tbody></table>
      <h3>EVENTS</h3>
      <table><tbody>
        <tr><th>INDUSTRY</th><th>EVENT TITLE</th><th>DATE</th></tr>
        <tr><td>Public Sector</td><td>MFA DIAL 2025 Day in a Life</td><td>9-10 Dec 2025</td></tr>
      </tbody></table>
      <p>Schneider Career Growth Days - 2025</p>
      <p>Airbus Fly Your Ideas 2026 student innovation challenge</p>
    </body></html>
    """

    email = EmailMessage(
        message_id="m6",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        body_html=html,
    )
    corpus, _, _, warnings = batch_sources.build_source_corpus(email, fetch_linked_pdfs=False)

    def no_llm(_chunk: str):
        raise AssertionError("job table and event-only residual should not require LLM")

    monkeypatch.setattr(all_job_extraction, "_invoke", no_llm)
    opportunities, metrics, errors = all_job_extraction.extract_all_opportunities(
        source_name="Goh Ze Li",
        source_message_id="m6",
        source_date=None,
        corpus=corpus,
    )

    pairs = {(item.company, item.role_title): item for item in opportunities}
    assert set(pairs) == {
        ("OPPO", "Site Reliability Engineer"),
        ("OPPO", "[Qianhai] Intelligent Manufacturing Engineer (Process Direction)"),
        ("OPPO", "[Qianhai] Software Product Manager (Overseas)"),
    }
    assert pairs[("OPPO", "Site Reliability Engineer")].location == "Singapore"
    assert pairs[("OPPO", "[Qianhai] Intelligent Manufacturing Engineer (Process Direction)")].location == "Shenzhen, China"
    assert pairs[("OPPO", "[Qianhai] Software Product Manager (Overseas)")].location == "Shenzhen, China"
    assert metrics.llm_calls == 0
    assert warnings == []
    assert errors == []


def test_ordinal_deadline_is_parsed_for_expiry_gate() -> None:
    assert str(all_job_extraction._parse_date_hint("Deadline: 5th Dec 2025")) == "2025-12-05"
    assert str(all_job_extraction._parse_date_hint("Deadline: 21st Dec 25")) == "2025-12-21"
    assert str(all_job_extraction._parse_date_hint("Deadline: 23rd Dec 2025")) == "2025-12-23"
    assert str(all_job_extraction._parse_date_hint("Deadline: 24th Dec 25")) == "2025-12-24"


def test_pdf_embedded_direct_urls_are_reattached_to_matching_roles(monkeypatch) -> None:
    internship_url = (
        "https://careers.point72.com/CSJobDetail?"
        "jobName=point72-academy-investment-analyst-summer-internship-2027-sg&"
        "jobCode=CPA-0015001"
    )
    full_time_url = (
        "https://careers.point72.com/CSJobDetail?"
        "jobName=point72-academy-investment-analyst-program-for-upcoming-graduates-2027-sg&"
        "jobCode=CPA-0014976"
    )

    def fake_invoke(_chunk: str) -> ExtractedOpportunityBatch:
        return ExtractedOpportunityBatch(
            opportunities=[
                ExtractedOpportunity(
                    company="Point72",
                    role_title="Point72 Academy Investment Analyst Summer Internship",
                    opportunity_type="internship",
                    evidence_text="Point72 Academy Investment Analyst Summer Internship",
                ),
                ExtractedOpportunity(
                    company="Point72",
                    role_title="Point72 Academy Investment Analyst Program for Upcoming Graduates",
                    opportunity_type="full_time",
                    evidence_text="Point72 Academy Investment Analyst Program for Upcoming Graduates",
                ),
            ]
        )

    monkeypatch.setattr(all_job_extraction, "_invoke", fake_invoke)
    corpus = (
        "SOURCE: LINKED PDF PAGE\nPAGE: 1\n"
        "Point72 Academy Investment Analyst opportunities\n"
        "[[SIMPLYNEXT_PDF_PAGE_LINKS]]\n"
        f"<{internship_url}>\n<{full_time_url}>"
    )

    opportunities, _, errors = all_job_extraction.extract_all_opportunities(
        source_name="TalentConnect",
        source_message_id="m7",
        source_date=None,
        corpus=corpus,
    )

    by_title = {item.role_title: item for item in opportunities}
    assert by_title["Point72 Academy Investment Analyst Summer Internship"].urls == [internship_url]
    assert by_title["Point72 Academy Investment Analyst Program for Upcoming Graduates"].urls == [full_time_url]
    assert errors == []
