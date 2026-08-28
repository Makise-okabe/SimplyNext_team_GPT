from career_agent.graph.workflow import career_agent_workflow
from career_agent.nodes.verify_job import verify_job


def test_workflow_stops_for_unrelated_email() -> None:
    result = career_agent_workflow.invoke(
        {
            "email": {
                "message_id": "x",
                "sender_email": "friend@example.com",
                "subject": "hello",
                "body_text": "not a career email",
            },
            "errors": [],
        }
    )

    assert result["is_career_email"] is False
    assert not result.get("verified_jobs")


def test_verify_job_marks_matching_public_page_verified() -> None:
    state = {
        "candidate_jobs": [
            {
                "company": "AMD",
                "title": "Product Development Engineer",
                "location": "Singapore",
                "opportunity_type": "full_time",
                "official_url": "https://careers.amd.com/job/123",
                "evidence": ["AMD Product Development Engineer"],
            }
        ],
        "resolved_pages": [
            {
                "requested_url": "https://careers.amd.com/job/123",
                "final_url": "https://careers.amd.com/job/123",
                "status_code": 200,
                "title": "Product Development Engineer | AMD Careers",
                "text": "AMD is hiring a Product Development Engineer in Singapore.",
            }
        ],
        "errors": [],
    }

    result = verify_job(state)

    job = result["verified_jobs"][0]
    assert job["verification_status"] == "verified"
    assert job["verification_basis"] == "official_web"


def test_verify_job_is_conservative_without_page() -> None:
    result = verify_job(
        {
            "candidate_jobs": [
                {
                    "company": "Example",
                    "title": "Engineer",
                    "opportunity_type": "unknown",
                    "official_url": "https://example.com/job",
                }
            ],
            "resolved_pages": [],
            "errors": [],
        }
    )

    job = result["verified_jobs"][0]
    assert job["verification_status"] == "partial"
    assert job["verification_basis"] == "public_web"


def test_verify_job_can_source_verify_trusted_pdf_attachment() -> None:
    result = verify_job(
        {
            "email": {
                "message_id": "mckinsey-mail",
                "sender_name": "Goh Ze Li",
                "sender_email": "zeli.goh@nus.edu.sg",
                "attachment_text": (
                    "Role: Innovation and Learning Centre ILC Intern. "
                    "McKinsey & Company is hiring this intern in Singapore."
                ),
            },
            "candidate_jobs": [
                {
                    "company": "McKinsey & Company",
                    "title": "Innovation and Learning Centre (ILC) Intern",
                    "opportunity_type": "internship",
                    "official_url": None,
                }
            ],
            "resolved_pages": [],
            "errors": [],
        }
    )

    job = result["verified_jobs"][0]
    assert job["verification_status"] == "source_verified"
    assert job["verification_basis"] == "trusted_email_attachment"
