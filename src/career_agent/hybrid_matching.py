from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


SKILL_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "c/c++": ("c++", "cplusplus", "embedded c", " c "),
    "java": ("java",),
    "programming": ("programming", "coding"),
    "javascript/typescript": ("javascript", "typescript", "node.js", "nodejs", "react.js", "reactjs"),
    "sql": ("sql", "text to sql", "database query"),
    "data analysis": ("data analysis", "data analytics", "analytics", "analyst"),
    "machine learning": ("machine learning", "ml engineer", "ai engineer", "artificial intelligence"),
    "deep learning": ("deep learning", "neural network", "pytorch", "tensorflow"),
    "generative ai": ("generative ai", "genai", "large language model", "llm"),
    "computer vision": ("computer vision", "image processing"),
    "nlp": ("natural language processing", "nlp"),
    "cloud": ("cloud", "aws", "azure", "gcp"),
    "devops": ("devops", "ci/cd", "site reliability", "sre", "kubernetes", "docker"),
    "software engineering": ("software engineer", "software engineering", "software development", "developer", "full stack", "full-stack", "backend", "front-end", "frontend"),
    "embedded systems": ("embedded", "firmware", "microcontroller", "mcu", "rtos"),
    "digital design": ("digital design", "digital ic", "logic design", "rtl"),
    "verilog/hdl": ("verilog", "systemverilog", "vhdl", "hdl", "rtl"),
    "fpga": ("fpga",),
    "analog circuits": ("analog", "analog ic", "circuit design", "electronics engineer", "electronics engineering"),
    "semiconductor": ("semiconductor", "chip design", "microelectronics", "wafer", "device engineering"),
    "eda/cadence": ("cadence", "virtuoso", "spectre", "eda", "electronic design automation"),
    "signal processing": ("signal processing", "signals and systems", "dsp"),
    "rf/wireless": ("rf", "wireless", "communications engineer", "communication systems"),
    "iot": ("internet of things", "iot"),
    "cybersecurity": ("cybersecurity", "cyber security", "security engineer", "vulnerability", "threat researcher"),
    "automation": ("automation", "rpa", "power automate"),
    "excel": ("excel",),
    "power bi": ("power bi",),
    "power systems": ("power systems", "power electronics", "electrical energy systems"),
    "photonics": ("photonics", "optical", "optoelectronics"),
    "financial modelling": ("financial model", "financial modelling", "financial modeling", "valuation"),
    "investment research": ("investment analyst", "equity research", "investment research", "fund management"),
    "quantitative analysis": ("quantitative", "quant research", "statistics", "statistical"),
    "project management": ("project management", "project engineer", "project planning", "scheduling"),
    "product management": ("product manager", "product management"),
    "stakeholder management": ("stakeholder", "cross-functional", "client management"),
    "communication": ("communication", "communications", "presentation", "presenting"),
    "leadership": ("leadership", "management associate", "graduate programme", "graduate program"),
    "marketing": ("marketing", "brand", "crm", "trade marketing"),
    "sales": ("sales", "business development"),
    "supply chain": ("supply chain", "procurement", "demand planner", "logistics"),
    "accounting": ("accounting", "accountant", "tax"),
    "mechanical design": ("mechanical design", "mechanical engineer"),
    "quality/reliability": ("reliability", "quality engineer", "failure analysis"),
}


TITLE_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(ai|machine learning|ml)\b", re.I), ("python", "machine learning", "deep learning")),
    (re.compile(r"\bdata (analyst|science|scientist|intelligence)\b", re.I), ("python", "sql", "data analysis")),
    (
        re.compile(r"\b(backend|full[- ]?stack|software engineer|software development engineer|developer|front[- ]?end)\b", re.I),
        ("software engineering", "programming"),
    ),
    (re.compile(r"\b(site reliability|sre|devops)\b", re.I), ("devops", "cloud", "software engineering", "programming")),
    (re.compile(r"\b(embedded|firmware)\b", re.I), ("embedded systems", "c/c++", "digital design")),
    (re.compile(r"\b(chip|ic design|semiconductor)\b", re.I), ("semiconductor", "digital design", "analog circuits", "eda/cadence")),
    (
        re.compile(r"\b(electrical|electronics|electronic|hardware)\b", re.I),
        ("analog circuits", "digital design", "embedded systems", "signal processing"),
    ),
    (re.compile(r"\b(power electronics|power engineer|power systems?)\b", re.I), ("power systems", "analog circuits")),
    (re.compile(r"\b(fpga|rtl|verilog|digital design)\b", re.I), ("digital design", "verilog/hdl", "fpga")),
    (re.compile(r"\b(photonics|optical|optoelectronic)\b", re.I), ("photonics", "semiconductor")),
    (re.compile(r"\b(cyber|security|vulnerability|threat)\b", re.I), ("cybersecurity", "software engineering", "programming")),
    (re.compile(r"\b(quant|investment analyst|fund management|global markets)\b", re.I), ("quantitative analysis", "investment research", "financial modelling")),
    (re.compile(r"\b(project engineer|project management)\b", re.I), ("project management", "stakeholder management", "communication")),
    (re.compile(r"\b(management associate|graduate programme|graduate program)\b", re.I), ("leadership", "communication", "stakeholder management")),
    (re.compile(r"\b(product manager|product management)\b", re.I), ("product management", "stakeholder management", "communication")),
    (re.compile(r"\b(field application engineer|applications? engineer)\b", re.I), ("stakeholder management", "communication")),
    (re.compile(r"\b(marketing|brand|crm)\b", re.I), ("marketing", "communication", "data analysis")),
    (re.compile(r"\b(procurement|supply chain|demand planner|logistics)\b", re.I), ("supply chain", "data analysis", "communication")),
    (re.compile(r"\b(accountant|accounting|tax)\b", re.I), ("accounting", "excel", "data analysis")),
)

ALIASES = {
    "git": "software engineering",
}


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _contains_phrase(value: str, phrase: str) -> bool:
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    if re.fullmatch(r"[a-z0-9]+", phrase):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", value))
    return phrase in value


def extract_explicit_skills(text: str | None) -> set[str]:
    value = f" {_normalize_text(text)} "
    found: set[str] = set()
    for skill, phrases in SKILL_PATTERNS.items():
        if any(_contains_phrase(value, phrase) for phrase in phrases):
            found.add(skill)
    return found


def infer_title_skills(title: str | None) -> set[str]:
    value = title or ""
    skills = extract_explicit_skills(value)
    for pattern, inferred in TITLE_RULES:
        if pattern.search(value):
            skills.update(ALIASES.get(skill, skill) for skill in inferred)
    return skills


def normalize_skills(skills: Iterable[str]) -> set[str]:
    return {ALIASES.get(skill.strip().lower(), skill.strip().lower()) for skill in skills if skill and skill.strip()}


@dataclass(frozen=True)
class JobSkillProfile:
    company: str
    title: str
    skills: tuple[str, ...]
    evidence_level: str
    evidence_text: str
    inferred_skills: tuple[str, ...]


@dataclass(frozen=True)
class StudentSkillProfile:
    skills: tuple[str, ...]
    raw_text: str


@dataclass(frozen=True)
class MatchResult:
    company: str
    title: str
    score: float
    confidence: str
    evidence_level: str
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    job_skills: tuple[str, ...]


def build_job_skill_profile(job: dict) -> JobSkillProfile:
    company = str(job.get("company") or "")
    title = str(job.get("title") or "")
    evidence_level = str(job.get("matching_evidence_level") or "source_only")
    jd_text = str(job.get("jd_text") or "")
    source_evidence = str(job.get("source_evidence") or "")
    matching_input = str(job.get("matching_input_text") or "")

    if evidence_level == "full_jd" and jd_text.strip():
        evidence_text = jd_text
    elif source_evidence.strip():
        evidence_text = source_evidence
    else:
        evidence_text = matching_input or f"{company} {title}"

    explicit = extract_explicit_skills(evidence_text)
    title_inferred = infer_title_skills(title)
    skills = explicit | title_inferred
    inferred_only = title_inferred - explicit

    return JobSkillProfile(
        company=company,
        title=title,
        skills=tuple(sorted(skills)),
        evidence_level=evidence_level,
        evidence_text=evidence_text,
        inferred_skills=tuple(sorted(inferred_only)),
    )


def build_student_skill_profile(
    *,
    resume_text: str = "",
    transcript_text: str = "",
    course_skill_texts: Iterable[str] = (),
    extra_skills: Iterable[str] = (),
) -> StudentSkillProfile:
    combined = "\n".join([resume_text, transcript_text, *course_skill_texts])
    skills = extract_explicit_skills(combined) | normalize_skills(extra_skills)
    return StudentSkillProfile(skills=tuple(sorted(skills)), raw_text=combined)


def score_match(student: StudentSkillProfile, job: JobSkillProfile) -> MatchResult:
    student_skills = set(student.skills)
    job_skills = set(job.skills)
    matched = student_skills & job_skills
    missing = job_skills - student_skills

    if job_skills:
        coverage = len(matched) / len(job_skills)
        score = round(100.0 * coverage, 1)
    else:
        score = 0.0

    if job.evidence_level == "full_jd":
        confidence = "high"
    elif len(job.evidence_text) >= 120 or len(job_skills) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return MatchResult(
        company=job.company,
        title=job.title,
        score=score,
        confidence=confidence,
        evidence_level=job.evidence_level,
        matched_skills=tuple(sorted(matched)),
        missing_skills=tuple(sorted(missing)),
        job_skills=tuple(sorted(job_skills)),
    )
