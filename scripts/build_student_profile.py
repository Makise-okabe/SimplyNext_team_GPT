from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.student_profile import build_student_profile

DEFAULT_OUTPUT = Path("data/student_profiles/latest_student_profile.json")


def _pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reusable student skill profile from a resume PDF and transcript PDF."
    )
    parser.add_argument("--resume", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--no-course-enrichment",
        action="store_true",
        help="Skip NUSMods module-description enrichment.",
    )
    args = parser.parse_args()

    resume_path = Path(args.resume)
    transcript_path = Path(args.transcript)
    if not resume_path.exists():
        raise FileNotFoundError(resume_path)
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    resume_text = _pdf_text(resume_path)
    transcript_text = _pdf_text(transcript_path)
    profile = build_student_profile(
        resume_text=resume_text,
        transcript_text=transcript_text,
        enrich_modules=not args.no_course_enrichment,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "simplinext.student_profile.v1",
        "resume_file": resume_path.name,
        "transcript_file": transcript_path.name,
        **profile.to_dict(),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Resume chars          :", profile.resume_text_chars)
    print("Transcript chars      :", profile.transcript_text_chars)
    print("Modules detected      :", len(profile.module_codes))
    print("Explicit skills       :", len(profile.explicit_skills))
    print("Course-derived skills :", len(profile.course_derived_skills))
    print("Total skills          :", len(profile.all_skills))
    print("Output                :", output)


if __name__ == "__main__":
    main()
