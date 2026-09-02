from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.matching_dataset import (
    is_matching_candidate,
    matching_evidence_level,
    matching_input_text,
    sanitize_job_sources,
)
from career_agent.models.job_record import JobRecord

DEFAULT_INPUT = Path("data/job_records/latest_job_catalog.json")
DEFAULT_OUTPUT = Path("data/job_records/latest_matching_candidates.json")
STALE_RESEARCH_BASES = {"trusted_nus_email_web_unresolved"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export active canonical jobs for resume/transcript matching."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Allow export from a catalog produced by the retired pre-fast-path research logic.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run the Track B catalog pipeline first."
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    raw_jobs = payload.get("jobs", [])

    stale_count = sum(
        raw.get("research_basis") in STALE_RESEARCH_BASES
        for raw in raw_jobs
    )
    if stale_count and not args.allow_stale:
        raise RuntimeError(
            f"Catalog is stale: {stale_count} job(s) were produced by the retired "
            "pre-fast-path web research logic. Run `uv run python "
            "scripts/run_all_job_research.py --scan 30` first, then export again."
        )

    candidates = []
    for raw in raw_jobs:
        clean_raw = dict(raw)
        clean_raw.pop("matching_ready", None)
        clean_raw.pop("matching_candidate", None)
        clean_raw.pop("matching_evidence_level", None)
        job = sanitize_job_sources(JobRecord.model_validate(clean_raw))
        if not is_matching_candidate(job):
            continue
        item = job.model_dump(mode="json")
        item["matching_evidence_level"] = matching_evidence_level(job)
        item["matching_input_text"] = matching_input_text(job)
        candidates.append(item)

    full_jd = sum(item["matching_evidence_level"] == "full_jd" for item in candidates)
    source_only = sum(item["matching_evidence_level"] == "source_only" for item in candidates)

    output_payload = {
        "schema": "simplinext.matching_candidates.v1",
        "purpose": "Input for resume/transcript matching; source-only jobs are retained with lower evidence.",
        "job_count": len(candidates),
        "full_jd_count": full_jd,
        "source_only_count": source_only,
        "jobs": candidates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Matching candidates:", len(candidates))
    print("Full JD            :", full_jd)
    print("Source only        :", source_only)
    print("Output             :", output_path)


if __name__ == "__main__":
    main()
