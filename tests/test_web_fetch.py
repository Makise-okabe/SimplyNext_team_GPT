from __future__ import annotations

import httpx

from career_agent.tools.web_fetch import _ats_posting


MCF_URL = (
    "https://www.mycareersfuture.gov.sg/job/engineering/"
    "full-stack-java-developer-conex-healthcare-071e6038cf90444da907ff35beee6cae"
)


class FakeClient:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.urls: list[str] = []

    def get(self, url: str):
        self.urls.append(url)
        return self.response


def test_mcf_dynamic_page_uses_public_detail_api() -> None:
    endpoint = "https://api.mycareersfuture.gov.sg/v2/jobs/071e6038cf90444da907ff35beee6cae"
    response = httpx.Response(
        200,
        json={
            "uuid": "071e6038cf90444da907ff35beee6cae",
            "title": "Full Stack Java Developer",
            "description": "<p>Responsibilities</p><p>Build Java healthcare applications.</p>" * 4,
            "postedCompany": {"name": "CONEX HEALTHCARE PTE. LTD."},
            "employmentTypes": [{"employmentType": "Full Time"}],
            "metadata": {"dates": {"expiry": "2099-09-30"}},
        },
        request=httpx.Request("GET", endpoint),
    )
    client = FakeClient(response)

    posting = _ats_posting(client, MCF_URL)

    assert client.urls == [endpoint]
    assert posting["title"] == "Full Stack Java Developer"
    assert posting["hiringOrganization"]["name"] == "CONEX HEALTHCARE PTE. LTD."
    assert posting["validThrough"] == "2099-09-30"


def test_mcf_missing_detail_is_marked_closed() -> None:
    endpoint = "https://api.mycareersfuture.gov.sg/v2/jobs/071e6038cf90444da907ff35beee6cae"
    response = httpx.Response(
        404,
        json={"message": "UUID is not found in the database."},
        request=httpx.Request("GET", endpoint),
    )

    posting = _ats_posting(FakeClient(response), MCF_URL)

    assert posting["validThrough"] == "1970-01-01"
    assert "no longer available" in posting["description"]
