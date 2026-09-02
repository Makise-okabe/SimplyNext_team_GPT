from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from career_agent.nusmods_database import (
    DEFAULT_ACADEMIC_YEARS,
    DEFAULT_DB_PATH,
    build_database,
    database_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local SQLite database of NUSMods course metadata and skill tags."
    )
    parser.add_argument(
        "--academic-year",
        action="append",
        dest="academic_years",
        help="Academic year to import, e.g. 2026-2027. Repeat for multiple years.",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    years = tuple(args.academic_years or DEFAULT_ACADEMIC_YEARS)
    db_path = Path(args.db)
    print("Academic years:", ", ".join(years))
    print("Database      :", db_path)
    print("Downloading NUSMods moduleInfo datasets...")

    counts = build_database(
        academic_years=years,
        db_path=db_path,
        refresh=args.refresh,
    )
    stats = database_stats(db_path)

    print("\nNUSMODS DATABASE SUMMARY")
    for year in years:
        print(f"  {year}: {counts.get(year, 0)} course rows")
    print("Total rows          :", stats["rows"])
    print("Unique module codes :", stats["unique_module_codes"])
    print("Database            :", db_path)


if __name__ == "__main__":
    main()
