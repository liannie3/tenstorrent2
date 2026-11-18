"""Host-only contract test for the rag_bench-compatible TT-NN adapter."""

from __future__ import annotations

import csv
import importlib.util
import io
import struct
import sys
import tempfile
from pathlib import Path


def load_adapter(repo: Path):
    tools = repo / "tools"
    sys.path.insert(0, str(tools))
    path = tools / "ttnn_bench.py"
    spec = importlib.util.spec_from_file_location("ttnn_bench", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_bank(path: Path) -> None:
    entries = 2
    dimension = 2048
    keys = [0.0] * (entries * dimension)
    keys[0] = 1.0
    keys[dimension + 1] = 1.0
    offsets_start = 24 + len(keys) * 4
    payload_start = offsets_start + entries * 8
    payload_size = 4 + 1 + 101 * 1024 * 4
    with path.open("wb") as output:
        output.write(struct.pack("<8s4I", b"RAGBANK1", entries, dimension, 101, 1024))
        output.write(struct.pack(f"<{len(keys)}f", *keys))
        output.write(struct.pack("<2Q", payload_start, payload_start + payload_size))
        for description in (b"a", b"b"):
            output.write(struct.pack("<I", len(description)))
            output.write(description)
            output.write(bytes(101 * 1024 * 4))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    adapter = load_adapter(repo)

    class FakeSearcher:
        instances = []

        def __init__(self, _bank: Path, device: int):
            assert device == 0
            self.closed = False
            self.calls = 0
            self.instances.append(self)

        def search_batch(self, queries, query_count, threshold):
            assert len(queries) == query_count * 2048
            assert threshold == 0.7
            self.calls += 1
            matches = [
                adapter.Match(index % 2 == 0, index % 2, 0.8 if index % 2 == 0 else 0.6)
                for index in range(query_count)
            ]
            # The fake only needs the attributes consumed by the adapter.
            class Timing:
                pass
            return adapter.SearchResult(matches, Timing())

        def close(self):
            self.closed = True

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        bank = root / "bank.bin"
        queries = root / "queries.f32"
        expected = root / "expected.json"
        write_bank(bank)
        values = [0.0] * (4 * 2048)
        for index in range(4):
            values[index * 2048 + index % 2] = 1.0
        queries.write_bytes(struct.pack(f"<{len(values)}f", *values))
        expected.write_text("{}", encoding="utf-8")
        arguments = adapter.parse_arguments([
            str(bank), str(queries), str(expected),
            "--backend", "ttnn-reference",
            "--precision", "bf16",
            "--batch", "2",
            "--warmup", "1",
            "--iters", "2",
            "--threshold", "0.7",
        ])
        assert not arguments.tracy_signposts
        output = io.StringIO()
        adapter.run(arguments, output, FakeSearcher)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        assert list(rows[0]) == adapter.RAW_FIELDS
        assert [row["mode"] for row in rows[:5]] == [
            "bank_load", "query_load", "backend_init", "cold_start", "first_request"
        ]
        measured = [row for row in rows if row["mode"] == "measured"]
        assert len(measured) == 8
        assert {row["batch"] for row in measured} == {"2"}
        assert {row["query_index"] for row in measured} == {"0", "1", "2", "3"}
        assert all(row["device_timing_available"] == "0" for row in measured)
        summary = rows[-1]
        assert summary["mode"] == "measured_phase"
        assert summary["query_count"] == "8"
        assert summary["iteration"] == "4"
        assert FakeSearcher.instances[0].closed


if __name__ == "__main__":
    main()
