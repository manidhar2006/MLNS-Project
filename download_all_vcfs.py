#!/usr/bin/env python3
"""Download all sample VCFs listed in CRyPTIC reuse table.

Usage:
    python3 download_all_vcfs.py \
      --csv /path/to/CRyPTIC_reuse_table_20231208.csv \
      --out-dir /path/to/vcf \
      --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen

DEFAULT_TABLE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/"
    "reuse/CRyPTIC_reuse_table_20231208.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all CRyPTIC VCF files from reuse table")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CRyPTIC_reuse_table_20231208.csv",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to save .vcf.gz files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel downloads (default: 8)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per file on network failure (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout seconds per request (default: 120)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_TABLE_URL,
        help="Base table URL used to resolve relative VCF paths",
    )
    return parser.parse_args()


def load_vcf_jobs(csv_path: Path, base_url: str) -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ENA_SAMPLE", "VCF"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            ena_sample = (row.get("ENA_SAMPLE") or "").strip()
            vcf_rel = (row.get("VCF") or "").strip()
            if not ena_sample or not vcf_rel:
                continue
            if vcf_rel.upper() == "NA":
                continue

            url = urljoin(base_url, vcf_rel)
            jobs.append((ena_sample, url))

    return jobs


def stream_download(url: str, dest: Path, timeout: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def download_with_retry(
    sample: str,
    url: str,
    out_dir: Path,
    retries: int,
    timeout: int,
) -> Tuple[str, str]:
    dest = out_dir / f"{sample}.vcf.gz"

    # Skip completed files to support resume.
    if dest.exists() and dest.stat().st_size > 0:
        return sample, "SKIP"

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            stream_download(url, dest, timeout)
            return sample, "OK"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_err = exc
            # Small backoff to avoid hammering the server on transient failures.
            time.sleep(min(5, attempt))

    return sample, f"FAIL: {last_err}"


def print_progress(done: int, total: int, ok: int, skip: int, fail: int) -> None:
    pct = (done / total * 100.0) if total else 100.0
    print(
        f"[{done}/{total} | {pct:6.2f}%] OK={ok} SKIP={skip} FAIL={fail}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    jobs = load_vcf_jobs(csv_path, args.base_url)

    total = len(jobs)
    if total == 0:
        print("No VCF jobs found in CSV.")
        return 1

    print(f"Loaded {total} VCF download jobs")
    print(f"Output directory: {out_dir}")
    print(f"Workers: {args.workers} | Retries: {args.retries}")

    ok = 0
    skip = 0
    fail = 0
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_with_retry,
                sample,
                url,
                out_dir,
                args.retries,
                args.timeout,
            ): (sample, url)
            for sample, url in jobs
        }

        for fut in as_completed(futures):
            done += 1
            sample, status = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1
                print(f"FAIL {sample}: {status}", file=sys.stderr)

            if done % 25 == 0 or done == total:
                print_progress(done, total, ok, skip, fail)

    print("\nDone")
    print(f"OK:   {ok}")
    print(f"SKIP: {skip}")
    print(f"FAIL: {fail}")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
