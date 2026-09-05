"""Per-run public research reuse. No resume or mailbox content is cached here."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class ResearchSession:
    pages: dict = field(default_factory=dict)
    searches: dict = field(default_factory=dict)
    search_calls: int = 0
    fetch_calls: int = 0

    def fetch(self, url, fetcher):
        # Fragments can identify jobs in hash-routed ATS pages; keep them in keys.
        key = url
        if key not in self.pages:
            self.fetch_calls += 1
            try:
                self.pages[key] = fetcher(url, timeout_seconds=8.0)
            except Exception as exc:
                self.pages[key] = exc
        value = self.pages[key]
        if isinstance(value, Exception):
            raise value
        return value

    def search(self, query, searcher):
        if query not in self.searches:
            self.search_calls += 1
            try:
                self.searches[query] = searcher(query, max_results=8)
            except Exception:
                self.searches[query] = []
        return self.searches[query]


_CURRENT = ContextVar("simplynext_research_session", default=None)


def current_session():
    return _CURRENT.get() or ResearchSession()


@contextmanager
def research_session():
    session = ResearchSession()
    token = _CURRENT.set(session)
    try:
        yield session
    finally:
        _CURRENT.reset(token)
