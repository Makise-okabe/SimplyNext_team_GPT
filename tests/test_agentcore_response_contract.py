import json

import pytest

from career_agent.tools.web_search import _agentcore_search_results


@pytest.mark.parametrize("result", [
    {"isError": True, "content": [{"type": "text", "text": "Access denied"}]},
    {"content": [{"type": "text", "text": "not a search payload"}]},
])
def test_tool_failure_is_not_an_empty_search(result):
    with pytest.raises(RuntimeError):
        _agentcore_search_results({"result": result}, 8)


@pytest.mark.parametrize("structured", [True, False])
def test_documented_and_structured_results_keep_job_url(structured):
    body = {"results": [{"title": "Electrical Intern", "url": "https://www.bhglobal.com.sg/jobs/electrical-intern/", "text": "Electrical systems"}]}
    result = {"structuredContent": body} if structured else {
        "content": [{"type": "text", "text": "{}"}, {"type": "text", "text": json.dumps(body)}]
    }
    rows = _agentcore_search_results({"result": result}, 8)
    assert len(rows) == 1
    assert rows[0].url == body["results"][0]["url"]
    assert rows[0].snippet == "Electrical systems"


def test_valid_zero_result_response_is_distinct_from_failure():
    assert _agentcore_search_results({"result": {"structuredContent": {"results": []}}}, 8) == []
