from __future__ import annotations

from career_agent import talentconnect_extraction
from career_agent.models.signal import ExtractedOpportunity, ExtractedOpportunityBatch


def test_company_hiring_lead_without_exact_title_is_kept(monkeypatch) -> None:
    def fake_invoke(_chunk: str) -> ExtractedOpportunityBatch:
        return ExtractedOpportunityBatch(
            opportunities=[
                ExtractedOpportunity(
                    company="Mastercard",
                    role_title=None,
                    opportunity_type="unknown",
                    evidence_text="Mastercard career opportunities",
                ),
                ExtractedOpportunity(
                    company="Point72",
                    role_title="Point72 Academy Investment Analyst Summer Internship",
                    opportunity_type="internship",
                    evidence_text="Point72 Academy Investment Analyst Summer Internship",
                ),
            ]
        )

    monkeypatch.setattr(talentconnect_extraction, "_invoke", fake_invoke)

    opportunities, metrics, errors = talentconnect_extraction.extract_talentconnect_opportunities(
        source_name="TalentConnect",
        source_message_id="m1",
        source_date=None,
        corpus="SOURCE: LINKED PDF PAGE\n" + ("TalentConnect careers newsletter " * 20),
    )

    by_company = {item.company: item for item in opportunities}
    assert by_company["Mastercard"].role_title == "Career opportunities"
    assert by_company["Point72"].role_title == "Point72 Academy Investment Analyst Summer Internship"
    assert metrics.llm_calls == 1
    assert errors == []
