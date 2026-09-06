from __future__ import annotations

import httpx

from career_agent.tools.web_fetch import _ats_posting, _dynamic_career_links


MCF_URL = (
    "https://www.mycareersfuture.gov.sg/job/engineering/"
    "full-stack-java-developer-conex-healthcare-071e6038cf90444da907ff35beee6cae"
)


class FakeClient:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.urls: list[str] = []

    def get(self, url: str, **kwargs):
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


def test_lenovo_board_exposes_active_dynamic_detail_links() -> None:
    endpoint = (
        "https://talent.lenovo.com.cn/gateway/jobBase/list?"
        "currentPage=1&pageSize=100&projectType=3"
    )
    response = httpx.Response(
        200,
        json={"code": 0, "result": {"rows": [{
            "id": 2399,
            "jobName": "AI Solution Architect",
            "typeName": "Global Future Leaders",
            "firstDeptId": "SSG",
            "publishFlag": 1,
            "activateFlag": 1,
        }]}},
        request=httpx.Request("GET", endpoint),
    )
    client = FakeClient(response)

    links, labels = _dynamic_career_links(
        client,
        "https://talent.lenovo.com.cn/position?projectType=3",
    )

    detail = "https://talent.lenovo.com.cn/position/detail?id=2399"
    assert client.urls == [endpoint]
    assert links == (detail,)
    assert labels == ((detail, "AI Solution Architect Global Future Leaders SSG"),)


def test_lenovo_dynamic_detail_uses_official_job_api() -> None:
    endpoint = "https://talent.lenovo.com.cn/gateway/jobBase/list?jobId=2399"
    response = httpx.Response(
        200,
        json={"code": 0, "result": {"rows": [{
            "id": 2399,
            "jobName": "AI Solution Architect",
            "jobDuties": "<p>Develop and deploy RAG and LLM applications.</p>",
            "jobRequirement": "<p>Python, PyTorch or TensorFlow, Linux and Git.</p>",
            "workPlace": "Beijing",
            "publishFlag": 1,
            "activateFlag": 1,
        }]}},
        request=httpx.Request("GET", endpoint),
    )
    client = FakeClient(response)

    posting = _ats_posting(
        client,
        "https://talent.lenovo.com.cn/position/detail?id=2399",
    )

    assert client.urls == [endpoint]
    assert posting["title"] == "AI Solution Architect"
    assert posting["hiringOrganization"]["name"] == "Lenovo"
    assert "RAG and LLM" in posting["description"]
    assert posting["validThrough"] is None
