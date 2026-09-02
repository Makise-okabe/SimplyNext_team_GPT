from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

AGGREGATOR_HOST_MARKERS = (
    "linkedin.com",
    "indeed.",
    "glassdoor.",
    "jobstreet.",
    "jobsdb.",
    "trabajo.org",
    "talent.com",
    "grabjobs.",
    "foundit.",
    "jooble.",
    "builtin.com",
    "expertini.com",
)
ATS_HOST_MARKERS = (
    "myworkdayjobs.com",
    "workdayjobs.com",
    "workday.com",
    "greenhouse.io",
    "lever.co",
    "smartrecruiters.com",
    "successfactors.com",
    "taleo.net",
    "oraclecloud.com",
    "icims.com",
    "jobvite.com",
    "eightfold.ai",
    "mokahr.com",
)

CLOSED_MARKERS = (
    "no longer accepting applications",
    "this job is no longer available",
    "this position is no longer available",
    "job has expired",
    "position has been filled",
    "applications are closed",
)

STOP_MARKERS = (
    "\nsimilar jobs\n",
    "\npeople also viewed\n",
    "\nreferrals increase your chances",
    "\nget notified when a new job is posted",
    "\nget notified about new\n",
    "\nexplore top content on linkedin\n",
    "\nlinkedin\n©",
)

DROP_EXACT_LINES = {
    "skip to main content",
    "expand search",
    "jobs",
    "people",
    "learning",
    "clear text",
    "sign in",
    "join now",
    "save",
    "report this job",
    "show more",
    "show less",
    "email or phone",
    "password",
    "forgot password?",
}

JD_SIGNAL_GROUPS = (
    (
        "responsibilities",
        "job responsibilities",
        "key responsibilities",
        "what you'll do",
        "what you’ll do",
        "what you will do",
        "your role",
    ),
    (
        "requirements",
        "job requirements",
        "qualifications",
        "minimum qualifications",
        "preferred qualifications",
        "what we're looking for",
        "what we’re looking for",
        "who we're looking for",
        "who we’re looking for",
    ),
    (
        "job description",
        "about the role",
        "about this role",
        "position summary",
        "role overview",
    ),
    (
        "apply now",
        "apply for this job",
        "apply for this role",
        "employment type",
        "job function",
        "job type",
    ),
)


def host(url: str | None) -> str:
    if not url:
        return ""
    try:
        value = urlparse(url).netloc.lower().split(":", 1)[0]
    except ValueError:
        return ""
    return value[4:] if value.startswith("www.") else value


def is_aggregator_url(url: str | None) -> bool:
    value = host(url)
    return bool(value and any(marker in value for marker in AGGREGATOR_HOST_MARKERS))


def is_secondary_url(url: str | None) -> bool:
    return is_aggregator_url(url)


def _company_aliases(company: str | None) -> set[str]:
    raw = company or ""
    aliases: set[str] = set()

    for parenthetical in re.findall(r"\(([^)]{2,20})\)", raw):
        compact = re.sub(r"[^a-z0-9]", "", parenthetical.lower())
        if len(compact) >= 2:
            aliases.add(compact)

    all_tokens = re.findall(r"[a-z0-9]+", raw.lower())
    legal_stop = {
        "the", "pte", "ltd", "limited", "inc", "holdings", "singapore",
        "company", "private", "corporation", "corp", "solutions", "branch",
    }
    tokens = [token for token in all_tokens if token not in legal_stop]
    aliases.update(token for token in tokens if len(token) >= 3)

    acronym_tokens = [token for token in tokens if token not in {"and", "of", "asia"}]
    acronym = "".join(token[0] for token in acronym_tokens if token)
    if len(acronym) >= 2:
        aliases.add(acronym)

    compact = "".join(token for token in tokens if token != "group")
    if len(compact) >= 4:
        aliases.add(compact)
    return aliases


def _short_alias_matches_brand_host(alias: str, compact_host: str, host_tokens: set[str]) -> bool:
    if alias in host_tokens:
        return True

    career_markers = ("career", "careers", "jobs", "recruit")
    for token in host_tokens:
        if (
            len(alias) >= 2
            and token.startswith(alias)
            and any(marker in token for marker in career_markers)
        ):
            return True

    return (
        len(alias) >= 2
        and any(marker in compact_host for marker in career_markers)
        and (
            compact_host.startswith(alias)
            or any(f"{alias}{marker}" in compact_host for marker in career_markers)
        )
    )


def _company_identity_matches_url(url: str | None, company: str | None) -> bool:
    if not url or not company:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    raw = unquote(
        " ".join(
            [
                parsed.netloc.lower(),
                parsed.path.lower(),
                parsed.query.lower(),
                parsed.fragment.lower(),
            ]
        )
    )
    compact = re.sub(r"[^a-z0-9]", "", raw)
    tokens = set(re.findall(r"[a-z0-9]+", raw))

    for alias in _company_aliases(company):
        if len(alias) <= 3:
            if alias in tokens:
                return True
        elif alias in compact:
            return True
    return False


def is_plausible_official_url(url: str | None, company: str | None) -> bool:
    """Employer/ATS source gate with company-safe ATS validation."""
    value = host(url)
    if not value or is_aggregator_url(url):
        return False
    if value in {"careers.example.com", "jobs.example.com"}:
        return True

    if any(marker in value for marker in ATS_HOST_MARKERS):
        return _company_identity_matches_url(url, company)

    compact_host = re.sub(r"[^a-z0-9]", "", value)
    host_tokens = set(re.findall(r"[a-z0-9]+", value))
    for alias in _company_aliases(company):
        if len(alias) <= 3:
            if _short_alias_matches_brand_host(alias, compact_host, host_tokens):
                return True
        elif alias in compact_host:
            return True
    return False


def page_is_closed(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    return any(marker in lowered for marker in CLOSED_MARKERS)


def looks_like_job_description(text: str) -> bool:
    """Require multiple independent employment/JD structure signals.

    Long product, documentation, support, or marketing pages can accidentally
    contain a role-title token and the word ``requirements``. One such marker is
    not enough to make the page a job description.
    """
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        return False
    matched_groups = sum(
        1 for markers in JD_SIGNAL_GROUPS if any(marker in lowered for marker in markers)
    )
    return matched_groups >= 2


def clean_jd_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lowered = normalized.lower()
    cut = len(normalized)
    for marker in STOP_MARKERS:
        index = lowered.find(marker)
        if index >= 0:
            cut = min(cut, index)
    normalized = normalized[:cut]

    kept: list[str] = []
    previous = None
    for raw in normalized.splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue
        lowered_line = line.lower()
        if lowered_line in DROP_EXACT_LINES:
            continue
        if lowered_line.startswith("by clicking continue to join or sign in"):
            continue
        if lowered_line.startswith("use ai to assess how you fit"):
            continue
        if lowered_line.startswith("get ai-powered advice"):
            continue
        if line == previous:
            continue
        kept.append(line)
        previous = line

    cleaned = "\n".join(kept).strip()
    if not looks_like_job_description(cleaned):
        return ""
    return cleaned
