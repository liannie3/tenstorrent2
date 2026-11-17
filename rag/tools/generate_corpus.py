"""Generate a byte-reproducible synthetic RAG bank and query corpus."""

from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
from pathlib import Path

from build_bank import TEXT_DIM, build

DEFAULT_ENTRIES = 3_400
DEFAULT_THRESHOLD = 0.7
DEFAULT_QUERY_COUNT = 128
DEFAULT_SWEEP = (8, 100, 1_000, 3_400, 10_000)


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


def make_queries(entries: int, query_count: int = DEFAULT_QUERY_COUNT) -> list[list[float]]:
    queries = []
    for query_index in range(query_count - 1):
        queries.append(normalize(raw_key(query_index % entries)))

    # Alternating signs are deliberately unrelated to the smooth synthetic keys.
    queries.append([1.0 if column % 2 else -1.0 for column in range(TEXT_DIM)])
    return queries


def expected_results(entries: int, queries: list[list[float]], threshold: float) -> list[dict]:
    results = []
    for query_index, query in enumerate(queries):
        stored_query = [float32(value) for value in query]
        squared_norm = sum(value * value for value in stored_query)
        inverse_norm = float32(1.0 / math.sqrt(squared_norm))
        normalized = [float32(value * inverse_norm) for value in stored_query]
        if query_index == len(queries) - 1:
            # Evaluate the single rejection query exhaustively. This keeps large-bank
            # generation O(entries * dimension), rather than O(queries * entries *
            # dimension), while retaining a genuine gate-boundary path.
            scores = (
                float32(sum(
                    left * right for left, right in zip(
                        normalized,
                        (float32(value) for value in normalize(raw_key(entry))),
                    )
                ))
                for entry in range(entries)
            )
            best_id, best_score = max(enumerate(scores), key=lambda item: item[1])
        else:
            best_id = query_index % entries
            key = [float32(value) for value in normalize(raw_key(best_id))]
            best_score = float32(sum(
                left * right for left, right in zip(normalized, key)
            ))
        results.append({
            "query_index": query_index,
            "top1_id": best_id,
            "accepted": best_score >= threshold,
            "score": best_score,
        })
    return results


def generate(
    output: Path,
    threshold: float = DEFAULT_THRESHOLD,
    entries: int = DEFAULT_ENTRIES,
    query_count: int = DEFAULT_QUERY_COUNT,
) -> None:
    if not 1 <= entries <= 1_000_000:
        raise ValueError("entries must be between 1 and 1,000,000")
    if query_count < 2:
        raise ValueError("query count must be at least 2")
    output.mkdir(parents=True, exist_ok=True)
    entry_count = entries
    queries = make_queries(entry_count, query_count)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        visual = temporary / "visual.f32"
        visual.write_bytes(bytes(101 * 1024 * 4))
        entries = temporary / "entries.jsonl"
        with entries.open("w", encoding="utf-8", newline="\n") as rows:
            for entry in range(entry_count):
                values = raw_key(entry)
                rows.write(json.dumps({
                    "description": f"Synthetic driving scene {entry}",
                    "text_key": values,
                    "visual_f32": visual.name,
                }, separators=(",", ":")) + "\n")
        build(
            entries,
            output / "bank.bin",
            {
                "generator": "generate_corpus.py",
                "dataset": "synthetic-v1",
                "entries": entry_count,
                "corpus_schema": "synthetic-v2-scalable",
            },
        )

    with (output / "queries.f32").open("wb") as query_file:
        for query in queries:
            query_file.write(struct.pack(f"<{TEXT_DIM}f", *query))

    expected = {
        "format": "RAG_EXPECTED1",
        "text_dim": TEXT_DIM,
        "query_count": len(queries),
        "threshold": threshold,
        "results": expected_results(entry_count, queries, threshold),
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
    parser.add_argument(
        "--sweep",
        help="comma-separated entry counts; writes each corpus below <output>/entries-N",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--entries",
        type=int,
        default=DEFAULT_ENTRIES,
        help="number of memory-bank entries (default: 3400 paper baseline)",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help="number of queries (default: 128)",
    )
    arguments = parser.parse_args()
    sweep = (
        [int(item) for item in arguments.sweep.split(",") if item.strip()]
        if arguments.sweep else []
    )
    if sweep:
        for entry_count in sweep:
            generate(
                arguments.output / f"entries-{entry_count}",
                arguments.threshold, entry_count, arguments.query_count,
            )
    else:
        generate(
            arguments.output, arguments.threshold,
            arguments.entries, arguments.query_count,
        )
