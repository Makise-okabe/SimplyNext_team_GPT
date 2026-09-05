from types import SimpleNamespace

from career_agent.models.signal import ExtractedOpportunityBatch
from ui import talentconnect_cached as tc


class RateLimitError(Exception):
    def __init__(self, delay):
        super().__init__('429 rate limit')
        self.response = SimpleNamespace(headers={'retry-after': str(delay)})


def test_long_rate_limit_defers_uncached_chunks_but_recovers_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, 'CACHE_DIR', tmp_path)
    chunks = ['first ' * 30, 'second ' * 30, 'cached ' * 30]
    monkeypatch.setattr(tc, '_chunks', lambda _: chunks)
    tc._save_cached_batch(chunks[2], ExtractedOpportunityBatch(opportunities=[]))
    calls = []
    def invoke(chunk):
        calls.append(chunk)
        raise RateLimitError(3600)
    monkeypatch.setattr(tc, '_invoke', invoke)
    monkeypatch.setattr(tc.time, 'sleep', lambda _: (_ for _ in ()).throw(AssertionError('must not wait')))
    state = tc.ExtractionState()
    _, metrics, warnings = tc.extract_talentconnect_cached(
        source_name='TalentConnect', source_message_id='one', source_date=None,
        corpus='newsletter', state=state,
    )
    assert metrics.llm_calls == 1
    assert state.rate_limited
    assert any('deferred' in warning for warning in warnings)
    assert '1/3 reused' in warnings[0]
    assert not tc._cache_path(chunks[0]).exists()
    tc.extract_talentconnect_cached(source_name='TalentConnect', source_message_id='two',
        source_date=None, corpus='another newsletter', state=state)
    assert calls == [chunks[0]]


def test_short_retry_after_is_honoured_and_success_returned(monkeypatch):
    monkeypatch.setenv('SIMPLYNEXT_TC_MAX_ATTEMPTS', '2')
    monkeypatch.setenv('SIMPLYNEXT_TC_PACE_SECONDS', '0')
    calls = []
    waits = []
    batch = ExtractedOpportunityBatch(opportunities=[])
    def invoke(chunk):
        calls.append(chunk)
        if len(calls) == 1:
            raise RateLimitError(2)
        return batch
    monkeypatch.setattr(tc, '_invoke', invoke)
    monkeypatch.setattr(tc.time, 'sleep', waits.append)
    assert tc._invoke_with_retry('chunk') == (batch, 2, None)
    assert waits == [2]


def test_client_has_no_hidden_retry_loop(monkeypatch):
    from career_agent import talentconnect_extraction as extraction
    monkeypatch.setenv('GROQ_API_KEY', 'test')
    config = {}
    def client(**kwargs):
        config.update(kwargs)
        return SimpleNamespace(with_structured_output=lambda _: None)
    monkeypatch.setattr(extraction, 'ChatGroq', client)
    extraction._build_llm()
    assert config['max_retries'] == 0
    assert config['timeout'] == 20
