#!/usr/bin/env python3
"""Download the official CRyPTIC reuse phenotype CSV from EBI (June 2022 release)."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

DEFAULT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/"
    "reuse/CRyPTIC_reuse_table_20231208.csv"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "cryptic" / "CRyPTIC_reuse_table_20231208.csv",
    )
    args = p.parse_args()
    dest: Path = args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url} -> {dest}")
    with urlopen(args.url, timeout=120) as r, dest.open("wb") as out:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"Wrote {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
