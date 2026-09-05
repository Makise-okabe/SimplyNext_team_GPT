import json

import pytest

from career_agent.catalog_consolidation import consolidate_job_records
from career_agent.job_page_verifier import apply_page_verification, verify_job_page
from career_agent.models.job_record import JobRecord
from career_agent.presentation import actionable_job_links, job_card, verified_job_url
from career_agent.research_session import research_session
from career_agent.stage1_ranking import rank_jobs
from career_agent.stage2_ranking import _find_source_job
from career_agent.tools.web_fetch import parse_html_page

URL = 'https://reolink.com/careers/jobs/ai-engineer'


def job(**changes):
    return JobRecord(source_key='goh_ze_li', source_message_id='m1',
                     source_subject='Jobs', company='Reolink', title='AI Engineer',
                     opportunity_type='full_time', source_evidence='AI Engineer Python',
                     **changes)


def page(**changes):
    posting = {'@type': 'JobPosting', 'title': 'AI Engineer',
               'hiringOrganization': {'name': 'Reolink'},
               'description': '<h2>Responsibilities</h2>' + 'Build machine learning systems with Python. ' * 20,
               'jobLocation': {'address': {'addressCountry': 'SG'}}}
    posting.update(changes)
    return parse_html_page(URL, URL, 200, '<script type="application/ld+json">' + json.dumps({'@graph': [posting]}) + '</script>')


def test_structured_jd_survives_script_removal_and_export():
    record = apply_page_verification(job(), verify_job_page(job(), page()))
    card = job_card(record.model_dump(), {'application_url': 'https://wrong.example/form'})
    assert verified_job_url(card) == URL
    assert card['application_url'] == URL
    assert card['jd_source_url'] == URL
    assert 'Python' in card['jd_text']
    links = actionable_job_links(card)
    assert links[0] == ("Open official job ↗", URL)
    assert links[1][0] == "Search LinkedIn ↗"


@pytest.mark.parametrize('changes,status', [
    ({'title': 'Mechanical Engineer'}, 'wrong_role'),
    ({'title': 'Sr AI Engineer'}, 'wrong_role'),
    ({'hiringOrganization': {'name': 'Other Employer'}}, 'wrong_company'),
    ({'validThrough': '2001-01-01'}, 'closed'),
])
def test_destination_evidence_rejects_false_matches(changes, status):
    assert verify_job_page(job(), page(**changes)).status == status


def test_homepage_redirect_is_not_an_exact_job():
    fetched = page()
    from dataclasses import replace
    fetched = replace(fetched, final_url='https://reolink.com/')
    assert verify_job_page(job(), fetched).status == 'generic_page'


def test_old_alias_cannot_restore_unverified_button():
    card = job_card(job(application_url=URL).model_dump(), {'job_page_url': URL, 'link_verification_status': 'verified'})
    assert verified_job_url(card) is None
    assert card['application_url'] is None


def test_every_named_opportunity_has_clickable_search_fallbacks():
    card = job_card(job().model_dump(), {})
    links = actionable_job_links(card)
    assert links[0][0] == "Search LinkedIn ↗"
    assert links[0][1].startswith("https://www.linkedin.com/jobs/search/?")
    assert not any("google.com" in url for _, url in links)


def test_likely_official_candidate_is_shown_before_search_fallback():
    candidate = "https://careers.amd.com/jobs/98765"
    card = job_card(job(
        candidate_job_url=candidate,
        candidate_job_kind="official_candidate",
        candidate_job_reason="Search result matches company and title",
    ).model_dump(), {})
    links = actionable_job_links(card)
    assert links[0] == ("Check possible role page ↗", candidate)
    assert verified_job_url(card) is None


def test_ranking_cannot_reopen_closed_posting():
    record = apply_page_verification(job(), verify_job_page(job(), page()))
    record.availability_status = 'closed_by_official'
    card = job_card(record.model_dump(), {'availability_status': 'active_candidate'})
    assert verified_job_url(card) is None


def test_consolidation_keeps_jd_and_verified_url_together():
    verified = apply_page_verification(job(), verify_job_page(job(), page()))
    old = job(jd_status='fetched_official', jd_text='Old unverified content',
              jd_source_url='https://wrong.example/job', research_status='verified_exact_job',
              research_confidence='high', primary_source_url='https://wrong.example/job')
    merged = consolidate_job_records([old, verified])[0]
    assert merged.jd_source_url == merged.job_page_url == URL
    assert merged.jd_text == verified.jd_text


def test_same_title_different_ids_survive_ranking_lookup():
    records = consolidate_job_records([job(talentconnect_id='100'), job(talentconnect_id='200')])
    assert len(records) == 2
    payloads = [record.model_dump() for record in records]
    rankings = rank_jobs({'all_skills': ['Python']}, payloads)
    assert len({r.record_id for r in rankings}) == 2
    for ranking in rankings:
        assert _find_source_job(ranking.to_dict(), payloads)['record_id'] == ranking.record_id


def test_research_fetch_reused_only_within_run():
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return page()
    with research_session() as session:
        assert session.fetch(URL, fetch) is session.fetch(URL, fetch)
    with research_session() as session:
        session.fetch(URL, fetch)
    assert calls == [URL, URL]
