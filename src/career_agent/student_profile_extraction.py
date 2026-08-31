from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.models.student_profile import StudentProfile


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(StudentProfile)


def extract_student_profile(
    *,
    resume_text: str,
    transcript_text: str = "",
) -> StudentProfile:
    prompt = f"""
Create a factual StudentProfile from the supplied resume and transcript.

Rules:
- Never invent facts not present in the documents.
- Preserve explicit technical skills, tools, projects, experience, education, GPA,
  coursework/modules, languages, and dates when available.
- Coursework from the transcript should be listed by the names/codes actually present.
- Do not infer a skill merely because a course title sounds related.
- Do not infer work authorization, preferred roles, or preferred locations unless the
  documents explicitly state them.
- Keep project descriptions concise but preserve evidence useful for job matching.

RESUME:
{resume_text}

TRANSCRIPT:
{transcript_text}
""".strip()
    profile = _build_llm().invoke(prompt)
    return profile.model_copy(
        update={
            "raw_resume_text": resume_text,
            "raw_transcript_text": transcript_text,
        }
    )
