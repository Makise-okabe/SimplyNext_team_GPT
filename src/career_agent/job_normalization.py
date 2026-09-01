from __future__ import annotations

import re
from urllib.parse import unquote

from career_agent.models.signal import OpportunitySignal
from career_agent.nodes.normalize_email import extract_links_from_text

COMPANY_CANONICAL_ALIASES = {
    "pg": "procter gamble",
    "procter gamble": "procter gamble",
    "procterandgamble": "procter gamble",
    "watsons": "watsons",
    "deutschebank": "deutsche bank",
    "ey": "ernst young",
    "ernstyoung": "ernst young",
    "ernstyoungsingaporeey": "ernst young",
    "ernstyoungsolutions": "ernst young",
    "bostonconsultinggroup": "boston consulting group",
    "thebostonconsultinggroup": "boston consulting group",
}
LEGAL_SUFFIXES = {
    "pte", "ltd", "limited", "private", "inc", "corp", "corporation",
    "plc", "llp", "ag", "company", "co",
}
COMPANY_NOTE_PATTERN = re.compile(
    r"\s*\((?:more\s+roles?\s+on\s+(?:tc|talent\s*connect)|see\s+more\s+on\s+(?:tc|talent\s*connect))\)\s*$",
    re.I,
)
URL_PATTERN = re.compile(r"<?https?://[^>\s]+>?", re.I)
TRAILING_SEE_PATTERN = re.compile(r"\s+(?:see|see\s+attached|see\s+attachment)\s*$", re.I)

GENERIC_TITLE_TOKENS = {
    "job", "jobs", "role", "roles", "position", "positions", "career", "careers",
    "program", "programme", "programs", "programmes", "opportunity", "opportunities",
    "full", "time", "upcoming", "graduate", "graduates", "student", "students",
    "singapore", "summer", "winter", "spring", "fall", "autumn", "campus",
    "recruitment", "global", "see", "attached", "attachment",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
}

CSIT_COMBINED_TITLE = (
    "Cyber Security Vulnerability Researcher Cybersecurity Specialist Cyber Threat "
    "Researcher Mobile and Cloud Security Engineer Cybersecurity Software Engineering"
)
CSIT_ROLES = (
    "Cyber Security Vulnerability Researcher",
    "Cybersecurity Specialist",
    "Cyber Threat Researcher",
    "Mobile and Cloud Security Engineer",
    "Cybersecurity Software Engineering",
)


def clean_company_name(company: str | None) -> str | None:
    if not company:
        return company
    value = " ".join(company.split()).strip()
    value = COMPANY_NOTE_PATTERN.sub("", value).strip()
    return value or company


def clean_role_title(title: str | None) -> tuple[str | None, list[str]]:
    if not title:
        return title, []
    urls = list(dict.fromkeys(extract_links_from_text(title)))
    value = URL_PATTERN.sub("", title)
    value = value.replace("<>", " ")
    value = TRAILING_SEE_PATTERN.sub("", value)
    value = re.sub(r"\s+", " ", value).strip(" -|:;,")
    return (value or title.strip()), urls


def canonical_company_text(company: str | None) -> str:
    raw = (clean_company_name(company) or "unknown").lower().replace("&", " and ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens = [token for token in raw.split() if token not in LEGAL_SUFFIXES]
    value = " ".join(tokens).strip()
    compact = "".join(token for token in tokens if token != "and")
    if compact in COMPANY_CANONICAL_ALIASES:
        return COMPANY_CANONICAL_ALIASES[compact]
    return value.replace(" and ", " ").strip()


def _company_tokens(company: str | None) -> set[str]:
    return {token for token in canonical_company_text(company).split() if len(token) >= 2}


def title_core_tokens(title: str | None, company: str | None = None) -> set[str]:
    cleaned, _ = clean_role_title(title)
    tokens = set(re.findall(r"[a-z0-9]+", (cleaned or "").lower()))
    company_tokens = _company_tokens(company)
    return {
        token
        for token in tokens
        if token not in GENERIC_TITLE_TOKENS
        and token not in company_tokens
        and not re.fullmatch(r"20\d{2}", token)
    }


def titles_equivalent(
    left_title: str | None,
    right_title: str | None,
    *,
    left_company: str | None = None,
    right_company: str | None = None,
) -> bool:
    left, _ = clean_role_title(left_title)
    right, _ = clean_role_title(right_title)
    left_norm = re.sub(r"[^a-z0-9]+", " ", (left or "").lower()).strip()
    right_norm = re.sub(r"[^a-z0-9]+", " ", (right or "").lower()).strip()
    if left_norm == right_norm:
        return True

    left_core = title_core_tokens(left, left_company)
    right_core = title_core_tokens(right, right_company)
    if not left_core or not right_core:
        return False
    intersection = left_core & right_core
    union = left_core | right_core
    jaccard = len(intersection) / max(1, len(union))
    containment = len(intersection) / max(1, min(len(left_core), len(right_core)))
    if min(len(left_core), len(right_core)) == 1:
        return left_core == right_core and len(left_norm) >= 8 and len(right_norm) >= 8
    return jaccard >= 0.72 or containment >= 0.88


def normalize_signal(signal: OpportunitySignal) -> OpportunitySignal:
    company = clean_company_name(signal.company)
    title, title_urls = clean_role_title(signal.role_title)
    urls = list(dict.fromkeys([*signal.urls, *title_urls]))
    return signal.model_copy(
        update={
            "company": company,
            "role_title": title,
            "urls": urls,
        }
    )


def expand_known_multi_role_signal(signal: OpportunitySignal) -> list[OpportunitySignal]:
    normalized = normalize_signal(signal)
    company_key = canonical_company_text(normalized.company)
    title = " ".join((normalized.role_title or "").split())
    if company_key == "centre for strategic infocomm technologies" and title == CSIT_COMBINED_TITLE:
        return [
            normalized.model_copy(update={"role_title": role})
            for role in CSIT_ROLES
        ]
    return [normalized]


def dedupe_signals(signals: list[OpportunitySignal]) -> list[OpportunitySignal]:
    merged: list[OpportunitySignal] = []
    for original in signals:
        signal = normalize_signal(original)
        match_index = None
        for index, existing in enumerate(merged):
            if canonical_company_text(existing.company) != canonical_company_text(signal.company):
                continue
            if existing.opportunity_type != signal.opportunity_type and "unknown" not in {
                existing.opportunity_type,
                signal.opportunity_type,
            }:
                continue
            if titles_equivalent(
                existing.role_title,
                signal.role_title,
                left_company=existing.company,
                right_company=signal.company,
            ):
                match_index = index
                break
        if match_index is None:
            merged.append(signal)
            continue

        previous = merged[match_index]
        preferred_title = previous.role_title
        if len(signal.role_title or "") > len(previous.role_title or "") and not URL_PATTERN.search(signal.role_title or ""):
            preferred_title = signal.role_title
        merged[match_index] = previous.model_copy(
            update={
                "company": clean_company_name(previous.company) or clean_company_name(signal.company),
                "role_title": preferred_title,
                "location": previous.location or signal.location,
                "opportunity_type": previous.opportunity_type if previous.opportunity_type != "unknown" else signal.opportunity_type,
                "deadline_hint": previous.deadline_hint or signal.deadline_hint,
                "target_major": list(dict.fromkeys([*previous.target_major, *signal.target_major])),
                "target_degree_level": list(dict.fromkeys([*previous.target_degree_level, *signal.target_degree_level])),
                "urls": list(dict.fromkeys([*previous.urls, *signal.urls])),
                "raw_text": previous.raw_text if len(previous.raw_text or "") >= len(signal.raw_text or "") else signal.raw_text,
            }
        )
    return merged
