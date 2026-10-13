"""Validate RAG backends against committed FP32 expected results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_common import parse_backend, run_benchmark, write_csv


FIELDS = [
    "backend",
    "precision",
    "query_count",
    "top1_agreement_pct",
    "gate_agreement_pct",
    "mean_abs_score_error",
    "max_abs_score_error",
    "passed",
]


def evaluate_backend(
    executable: Path,
    bank: Path,
    queries: Path,
    expected_path: Path,
    backend_spec: str,
    *,
    required_top1: float = 100.0,
    required_gate: float = 100.0,
    score_tolerance: float = 5e-3,
) -> dict:
    expected_document = json.loads(expected_path.read_text(encoding="utf-8"))
    expected = expected_document["results"]
    records = run_benchmark(
        executable,
        bank,
        queries,
        expected_path,
        backend_spec,
        batch=1,
        warmup=0,
        iterations=1,
        threshold=float(expected_document["threshold"]),
    )
    measured = [record for record in records if record["mode"] == "measured"]
    if len(measured) != len(expected):
        raise RuntimeError(
            f"{backend_spec}: expected {len(expected)} results, got {len(measured)}"
        )

    top1_matches = 0
    gate_matches = 0
    errors = []
    for actual, reference in zip(measured, expected):
        top1_matches += int(int(actual["id"]) == int(reference["top1_id"]))
        actual_accepted = actual["accepted"] in {"1", "true", "True"}
        gate_matches += int(actual_accepted == bool(reference["accepted"]))
        errors.append(abs(float(actual["score"]) - float(reference["score"])))

    query_count = len(expected)
    top1 = 100.0 * top1_matches / query_count
    gate = 100.0 * gate_matches / query_count
    maximum_error = max(errors, default=0.0)
    backend, precision = parse_backend(backend_spec)
    passed = (
        top1 >= required_top1
        and gate >= required_gate
        and maximum_error <= score_tolerance
    )
    return {
        "backend": backend,
        "precision": precision,
        "query_count": query_count,
        "top1_agreement_pct": f"{top1:.6f}",
        "gate_agreement_pct": f"{gate:.6f}",
        "mean_abs_score_error": f"{sum(errors) / query_count:.9g}",
        "max_abs_score_error": f"{maximum_error:.9g}",
        "passed": str(passed).lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-bench", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--backends", default="cpu")
    parser.add_argument("--required-top1", type=float, default=100.0)
    parser.add_argument("--required-gate", type=float, default=100.0)
    parser.add_argument("--score-tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "results"
            / "rag_quality_gate.csv"
        ),
    )
    arguments = parser.parse_args()

    rows = [
        evaluate_backend(
            arguments.rag_bench,
            arguments.bank,
            arguments.queries,
            arguments.expected,
            backend.strip(),
            required_top1=arguments.required_top1,
            required_gate=arguments.required_gate,
            score_tolerance=arguments.score_tolerance,
        )
        for backend in arguments.backends.split(",")
        if backend.strip()
    ]
    write_csv(arguments.output, FIELDS, rows)
    failed = [row["backend"] for row in rows if row["passed"] != "true"]
    if failed:
        print("quality gate failed: " + ", ".join(failed))
        return 1
    print(f"quality gate passed; wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
