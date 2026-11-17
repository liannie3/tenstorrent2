"""Validate whether a RAG result directory is suitable for baseline reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--minimum-entries", type=int, default=3_400)
    parser.add_argument("--minimum-trials", type=int, default=10)
    parser.add_argument("--maximum-cv", type=float, default=5.0)
    parser.add_argument("--required-batches", default="1,4,16,32,128")
    args = parser.parse_args()

    errors: list[str] = []
    manifest = json.loads(
        (args.results / "rag_run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "rag-benchmark-manifest-v5":
        errors.append("manifest is not rag-benchmark-manifest-v5")
    if int(manifest.get("independent_trials", 0)) < args.minimum_trials:
        errors.append(f"fewer than {args.minimum_trials} independent trials")

    required_batches = {
        int(item) for item in args.required_batches.split(",") if item.strip()
    }
    actual_batches = {int(item) for item in manifest.get("batches", [])}
    missing_batches = required_batches - actual_batches
    if missing_batches:
        errors.append(f"missing batches: {sorted(missing_batches)}")

    memory = rows(args.results / "rag_memory_layer.csv")
    entry_counts = {int(row["Bank entries"]) for row in memory}
    if not entry_counts or min(entry_counts) < args.minimum_entries:
        errors.append(f"bank contains fewer than {args.minimum_entries} entries")

    quality = rows(args.results / "rag_quality_gate.csv")
    failed = [row["backend"] for row in quality if row["passed"] != "true"]
    if failed:
        errors.append(f"quality gate failed: {', '.join(failed)}")

    application = rows(args.results / "rag_application_layer.csv")
    unstable = [
        f"{row['Platform']}:{row['Type']}:b{row['Batch']}={row['Throughput CV %']}%"
        for row in application
        if row["Mode"].endswith("/all") and row["Throughput CV %"]
        and float(row["Throughput CV %"]) > args.maximum_cv
    ]
    if unstable:
        errors.append("throughput CV exceeds limit: " + ", ".join(unstable))

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: campaign satisfies baseline publication gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
