"""Download official ATUS files into data/raw/<year>/ and record them in data/raw/manifest.csv.

Usage:
    python scripts/download_atus.py --years 2022 2023
    python scripts/download_atus.py --years 2022 --register   # checksum user-provided files instead

If BLS blocks the download, place files manually in data/raw/<year>/ (from
https://www.bls.gov/tus/data.htm) and re-run with --register.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import DEFAULT_KINDS, FILE_PATTERNS, download_year, register_existing  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--kinds", nargs="+", default=DEFAULT_KINDS, choices=sorted(FILE_PATTERNS))
    ap.add_argument("--register", action="store_true", help="checksum existing files, no download")
    args = ap.parse_args()

    for year in args.years:
        if args.register:
            register_existing(args.raw_dir, year)
        else:
            download_year(year, args.raw_dir, args.kinds)


if __name__ == "__main__":
    main()
