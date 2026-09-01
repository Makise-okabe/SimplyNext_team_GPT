from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_all_job_research_help_starts_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_all_job_research.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        env={},
    )

    assert result.returncode == 0, result.stderr
    assert "SimplyNext Track B" in result.stdout
    assert "--scan" in result.stdout
