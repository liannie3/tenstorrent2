#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d -p . results-min-XXXXXXXX)
mode=("$@")
if (( $# == 0 )); then mode=(--text-only); fi
args=(--model models/tinyllama-chat --warmup 1 --runs 3 --max-new-tokens 8 "${mode[@]}")
echo "Results: $out"
python benchmark_latency.py "${args[@]}" --output "$out/latency.csv"
python benchmark_cpu.py "${args[@]}" --output "$out/cpu.csv"
python benchmark_gpu.py "${args[@]}" --tt-smi --device-interval-ms 500 --output "$out/card.csv"
