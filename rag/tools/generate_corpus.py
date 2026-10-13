"""Generate a byte-reproducible synthetic RAG bank and query corpus."""

from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
from pathlib import Path

from build_bank import TEXT_DIM, build


DEFAULT_ENTRIES = 8
DEFAULT_THRESHOLD = 0.7
DEFAULT_QUERY_COUNT = 10


def raw_key(entry: int) -> list[float]:
    return [
        math.sin((entry + 1) * (column + 1) * 0.013)
        + 0.5 * math.cos((entry + 3) * (column + 1) * 0.007)
        for column in range(TEXT_DIM)
    ]


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_queries(keys: list[list[float]]) -> list[list[float]]:
    queries = []
    for query_index in range(DEFAULT_QUERY_COUNT - 1):
        key = keys[query_index % len(keys)]
        queries.append([
            value + 0.01 * math.sin((query_index + 5) * (column + 1) * 0.017)
            for column, value in enumerate(key)
        ])

    # A deterministic low-similarity query exercises the threshold rejection path.
    queries.append([
        math.sin((column + 1) * 0.371) * math.cos((column + 1) * 0.119)
        for column in range(TEXT_DIM)
    ])
    return queries


def expected_results(
    keys: list[list[float]], queries: list[list[float]], threshold: float
) -> list[dict]:
    results = []
    for query_index, query in enumerate(queries):
        stored_query = [float32(value) for value in query]
        squared_norm = sum(value * value for value in stored_query)
        inverse_norm = float32(1.0 / math.sqrt(squared_norm))
        normalized = [float32(value * inverse_norm) for value in stored_query]
        scores = [
            float32(sum(left * right for left, right in zip(normalized, key)))
            for key in keys
        ]
        best_id = max(range(len(scores)), key=scores.__getitem__)
        results.append({
            "query_index": query_index,
            "top1_id": best_id,
            "accepted": scores[best_id] >= threshold,
            "score": scores[best_id],
        })
    return results


def generate(output: Path, threshold: float = DEFAULT_THRESHOLD) -> None:
    output.mkdir(parents=True, exist_ok=True)
    key_rows = [raw_key(entry) for entry in range(DEFAULT_ENTRIES)]
    normalized_keys = [normalize(values) for values in key_rows]
    stored_keys = [
        [float32(value) for value in key] for key in normalized_keys
    ]
    queries = make_queries(normalized_keys)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        visual = temporary / "visual.f32"
        visual.write_bytes(bytes(101 * 1024 * 4))
        entries = temporary / "entries.jsonl"
        with entries.open("w", encoding="utf-8", newline="\n") as rows:
            for entry, values in enumerate(key_rows):
                rows.write(json.dumps({
                    "description": f"Synthetic driving scene {entry}",
                    "text_key": values,
                    "visual_f32": visual.name,
                }, separators=(",", ":")) + "\n")
        build(
            entries,
            output / "bank.bin",
            {"generator": "generate_corpus.py", "dataset": "synthetic-v1"},
        )

    with (output / "queries.f32").open("wb") as query_file:
        for query in queries:
            query_file.write(struct.pack(f"<{TEXT_DIM}f", *query))

    expected = {
        "format": "RAG_EXPECTED1",
        "text_dim": TEXT_DIM,
        "query_count": len(queries),
        "threshold": threshold,
        "results": expected_results(stored_keys, queries, threshold),
    }
    (output / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bench_data",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    arguments = parser.parse_args()
    generate(arguments.output, arguments.threshold)
