"""End-to-end smoke test for the stdlib-only benchmark harness."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_benchmark_tools.py RAG_BENCH")
    rag_bench = Path(sys.argv[1]).resolve()
    rag_root = Path(__file__).resolve().parents[1]
    tools = rag_root / "tools"
    sys.path.insert(0, str(tools))
    from bench_harness import hardware_row

    synthetic = []
    for trial, device_us in (("1", "20"), ("2", "22")):
        for query_index in (0, 1):
            synthetic.append({
                "_trial": trial,
                "mode": "measured",
                "iteration": "0",
                "query_index": str(query_index),
                "device_timing_available": "1",
                "device_total_us": device_us,
                "h2d_us": "2",
                "conversion_us": "0",
                "gemm_us": "14",
                "reduction_us": "3",
                "d2h_us": "1",
                "h2d_bytes": str(2 * 2048 * 4),
                "d2h_bytes": "24",
                "cpu_normalization_us": "4",
                "buffer_preparation_us": "1",
                "host_result_construction_us": "1",
                "vector_retrieval_us": "30",
                "accepted": "1",
            })
    aggregated = hardware_row(synthetic, "cuda-optimized", "fp32", 2, 8)
    assert aggregated["Trials"] == 2
    assert aggregated["Samples"] == 2
    assert aggregated["Device us/query"] == "10.500000"
    assert aggregated["Transfer bytes/query"] == "8204.000"

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        data = root / "data"
        results = root / "results"
        subprocess.run(
            [sys.executable, str(tools / "generate_corpus.py"), "--output", str(data)],
            check=True,
        )
        generated = [data / "bank.bin", data / "queries.f32", data / "expected.json"]
        first_hashes = [hashlib.sha256(path.read_bytes()).digest() for path in generated]
        subprocess.run(
            [sys.executable, str(tools / "generate_corpus.py"), "--output", str(data)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert first_hashes == [
            hashlib.sha256(path.read_bytes()).digest() for path in generated
        ]
        subprocess.run(
            [
                sys.executable,
                str(tools / "bench_harness.py"),
                "--rag-bench",
                str(rag_bench),
                "--bank",
                str(data / "bank.bin"),
                "--queries",
                str(data / "queries.f32"),
                "--expected",
                str(data / "expected.json"),
                "--warmup",
                "1",
                "--iters",
                "2",
                "--trials",
                "2",
                "--batches",
                "1,128",
                "--results",
                str(results),
            ],
            check=True,
        )
        with (results / "rag_quality_gate.csv").open(newline="", encoding="utf-8") as source:
            quality = list(csv.DictReader(source))
        assert quality[0]["passed"] == "true"
        with (results / "rag_application_layer.csv").open(newline="", encoding="utf-8") as source:
            application = list(csv.DictReader(source))
        assert {row["Mode"] for row in application} == {
            "Host-facing/all", "Host-facing/accepted", "Host-facing/rejected",
            "Batch request/all",
        }
        assert {int(row["Batch"]) for row in application} == {1, 128}
        assert {int(row["Trials"]) for row in application} == {2}
        assert float(application[0]["P95 ms"]) > 0.0
        assert float(application[0]["Queries/s"]) > 0.0

        with (results / "rag_stage_layer.csv").open(newline="", encoding="utf-8") as source:
            stages = list(csv.DictReader(source))
        assert {row["Stage"] for row in stages} == {
            "Vector retrieval", "Payload retrieval"
        }
        assert all(float(row["P95 ms"]) >= 0.0 for row in stages)
        assert (results / "rag_memory_layer.csv").is_file()
        manifest_path = results / "rag_run_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "rag-benchmark-manifest-v3"
        assert manifest["instrumentation"] == {
            "cuda_event_timing": False,
            "nvtx": False,
        }
        with (results / "rag_hardware_layer.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            hardware = list(csv.DictReader(source))
        assert all(row["Device us/query"] == "" for row in hardware)
        raw_path = results / "raw" / "cpu-b1-run01.csv"
        assert raw_path.is_file()
        with raw_path.open(newline="", encoding="utf-8") as source:
            raw = list(csv.DictReader(source))
        measured = [row for row in raw if row["mode"] == "measured"]
        summary = next(row for row in raw if row["mode"] == "measured_phase")
        assert int(summary["peak_host_rss_bytes"]) > 0
        assert all(
            float(row["end_to_end_us"]) >= float(row["vector_retrieval_us"])
            for row in measured
        )
        assert all(
            float(row["payload_retrieval_us"]) == 0.0
            for row in measured if row["accepted"] == "0"
        )
        assert all(
            float(row["payload_retrieval_us"]) > 0.0
            for row in measured if row["accepted"] == "1"
        )
        assert all(row["device_timing_available"] == "0" for row in measured)
        assert all(row["device_total_us"] == "" for row in measured)

        report = root / "report.md"
        report.write_text(
            (rag_root / "docs" / "performance_evaluation.md").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(tools / "render_report.py"),
                "--report",
                str(report),
                "--results",
                str(results),
            ],
            check=True,
        )
        rendered = report.read_text(encoding="utf-8")
        assert rendered.startswith("# RAG Subsystem Benchmark Report\n")
        assert "See the raw CSV files for full precision." in rendered
        assert "| CPU reference | FP32 | 1 | 2 | 512 | Single request · all |" in rendered
        assert "| CPU reference | FP32 | 1 | accepted | Payload retrieval |" in rendered


if __name__ == "__main__":
    main()
