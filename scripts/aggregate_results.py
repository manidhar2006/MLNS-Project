#!/usr/bin/env python3
"""
scripts/aggregate_results.py — Collate per-arm CV results into metrics_table.csv.

Reads runs/<arm>/cv_summary.json for each arm and merges into a single
long-format CSV with columns: arm, drug, metric, mean, std.

Usage:
    python scripts/aggregate_results.py [--runs-dir ./runs] [--output metrics_table.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_cv_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def flatten_summary(arm: str, summary: dict) -> list[dict]:
    """Convert {drug: {metric: {mean, std}}} to list of rows."""
    rows = []
    for drug, metric_map in summary.items():
        if drug.startswith("_"):
            continue
        for metric, stats in metric_map.items():
            if metric in ("n_R", "n_S"):
                continue
            if isinstance(stats, dict):
                rows.append({
                    "arm": arm,
                    "drug": drug,
                    "metric": metric,
                    "mean": stats.get("mean", float("nan")),
                    "std": stats.get("std", float("nan")),
                })
            else:
                # Scalar value (fold-level result, not aggregated)
                rows.append({
                    "arm": arm,
                    "drug": drug,
                    "metric": metric,
                    "mean": float(stats),
                    "std": float("nan"),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--output", default="metrics_table.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    arms_found = sorted([p.name for p in runs_dir.iterdir() if p.is_dir() and p.name in ("A", "B", "C", "D")])

    if not arms_found:
        print(f"No arm subdirectories found in {runs_dir}")
        return

    all_rows = []
    for arm in arms_found:
        summary_path = runs_dir / arm / "cv_summary.json"
        if not summary_path.exists():
            print(f"[skip] {summary_path} not found")
            continue
        summary = load_cv_summary(summary_path)
        rows = flatten_summary(arm, summary)
        all_rows.extend(rows)
        print(f"Arm {arm}: {len(rows)} metric rows")

    if not all_rows:
        print("No results found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # Print AUROC pivot table for quick inspection
    auroc = df[df["metric"] == "AUROC"].copy()
    auroc["val"] = auroc.apply(lambda r: f"{r['mean']:.4f}±{r['std']:.4f}", axis=1)
    try:
        pivot = auroc.pivot(index="drug", columns="arm", values="val")
        print("\nAUROC (mean±std) per arm x drug:")
        print(pivot.to_string())
    except Exception:
        print(auroc[["arm", "drug", "mean", "std"]].to_string(index=False))


if __name__ == "__main__":
    main()
