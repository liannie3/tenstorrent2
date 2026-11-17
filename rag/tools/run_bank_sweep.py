"""Generate and benchmark a deterministic RAG memory-bank scaling sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Keep smoke/CI fixtures separate from the publication scaling campaign.  These
# sizes cover a tiny production run, a small bank, the paper-comparison
# baseline, and the large-bank stress case respectively.
DEFAULT_SIZES = "10,100,3400,10000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-bench", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--backends", default="cpu,cuda-optimized,cuda-optimized:bf16")
    parser.add_argument("--batches", default="1,4,16,32,128")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--telemetry-provider", default="none")
    parser.add_argument("--telemetry-power-rail", default="unspecified")
    parser.add_argument("--power-mode", default="unspecified")
    parser.add_argument("--clock-policy", default="unspecified")
    parser.add_argument("--cpu-governor", default="unspecified")
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="reuse previously generated entries-N corpus directories",
    )
    args = parser.parse_args()

    tools = Path(__file__).resolve().parent
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise SystemExit("--sizes must contain positive integers")

    if not args.skip_generation:
        subprocess.run([
            sys.executable, str(tools / "generate_corpus.py"),
            "--output", str(args.data_root),
            "--sweep", ",".join(map(str, sizes)),
        ], check=True)
    for size in sizes:
        data = args.data_root / f"entries-{size}"
        output = args.results_root / f"entries-{size}"
        command = [
            sys.executable, str(tools / "bench_harness.py"),
            "--rag-bench", str(args.rag_bench),
            "--bank", str(data / "bank.bin"),
            "--queries", str(data / "queries.f32"),
            "--expected", str(data / "expected.json"),
            "--backends", args.backends,
            "--batches", args.batches,
            "--trials", str(args.trials),
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--results", str(output),
            "--telemetry-provider", args.telemetry_provider,
            "--telemetry-power-rail", args.telemetry_power_rail,
            "--power-mode", args.power_mode,
            "--clock-policy", args.clock_policy,
            "--cpu-governor", args.cpu_governor,
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
