# CUDA RAG retrieval module

This repository starts the RAG portion of CONOPS.pdf project, which recreates (not 1-1) the pipeline in Chiu et al., *Edge-Efficient Vision-Language Models for Autonomous Driving Using Distillation and RAG-Based Connectors* (CVPR Workshops 2026). The detector, TinyLlama inference, CLIP inference, student connector, and VLM generation are integration points owned by the rest of the project. This module packs their precomputed features, performs top-1 cosine retrieval, applies the conditional gate, and returns the winning description and visual features.

## Contract with other pipeline stages

The bank builder takes JSONL with one object per memory entry:

```json
{"description":"A pedestrian is crossing ...","text_key":[2048 floats],"visual_f32":"features/0001.bin"}
```

`text_key` is the mean-pooled 2048-dimensional hidden state from the team's fine-tuned TinyLlama model, using the exact same tokenization, pooling, and prompt template as runtime queries. `visual_f32` is a little-endian, row-major float32 file of 101 × 1024 CLIP features (CLS plus 100 selected patches). Paths are relative to the JSONL file. The builder L2-normalizes keys. The runtime caller supplies a 2048-float query embedding made from the user question and YOLO detection string; the library normalizes it before search. Decide and record the exact concatenation template and patch selection rule with the other teams before generating the full bank.

The result is `{accepted, id, score}`. When `accepted` is true, `Bank::load(id)` returns the description and `[101,1024]` visual features. The connector team projects those features and assembles the VLM input. A rejected result must not be injected into the prompt.

## Build

The Jetson development target uses JetPack **6.2.1+b38** and Python **3.10.12**.
See [docs/development.md](docs/development.md) for the complete toolchain contract and
setup instructions.

```sh
cd rag
cmake -S . -B build -DRAG_ENABLE_CUDA=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Use `-DRAG_ENABLE_CUDA=OFF` for a CPU-only build. The CUDA build requires the CUDA Toolkit and a supported host compiler. The CUDA backend has been compiled here with CUDA Toolkit 13.4 and MSVC 19.44. If an older terminal cannot find `nvcc`, reopen it so it inherits the CUDA installer’s updated system `PATH`.

## Pack and query

```sh
python tools/build_bank.py entries.jsonl memory.bank --metadata bank_metadata.json
build/rag_query memory.bank query.f32 0.7 --cuda-optimized
```

`query.f32` contains exactly 2048 little-endian float32 values. The CLI defaults to threshold 0.7 and CPU search. `--cuda` selects the simple CUDA reference backend; `--cuda-optimized` selects persistent cuBLAS scoring with device-side top-1 selection and gating. Add `--bf16` to the optimized backend for BF16 keys and queries with FP32 accumulation. Keep one searcher alive across queries in an integrated process so the prepared bank remains on the GPU.

## Benchmark harness

The default benchmark bank contains **3,400 entries**, matching the paper-comparison
baseline. At roughly 412 KiB per full entry, expect a bank around 1.4 GB. Generate
that baseline and run the quality-gated benchmark with:

```sh
python tools/generate_corpus.py
build/rag_bench bench_data/bank.bin bench_data/queries.f32 \
  bench_data/expected.json --backend cpu --batch 1 --warmup 10 --iters 100
python tools/bench_harness.py \
  --rag-bench build/rag_bench \
  --bank bench_data/bank.bin \
  --queries bench_data/queries.f32 \
  --expected bench_data/expected.json \
  --backends cpu,cuda-reference,cuda-optimized,cuda-optimized:bf16 \
  --batches 1,2,4,8,16,32,64,128 \
  --trials 10 \
  --run-kind release
python tools/render_report.py \
  --run-dir results/$(date -u +%F)
```

Results are grouped into one UTC-dated run bundle:

```text
results/YYYY-MM-DD/
  release/       # authoritative application, startup, memory, and thermal data
  cuda-timing/   # CUDA-event hardware measurements
  profiles/      # Nsight reports and exported summaries
  report/        # combined Markdown report and charts
```

Omit `--results` to use this layout automatically. Select `--run-kind release`,
`--run-kind cuda-timing`, or `--run-kind smoke`. Use `--run-date YYYY-MM-DD` when
reproducing or importing a run for a specific date. An explicit `--results` remains
available for tests and one-off experiments.

Pass `--entries N` to generate a different memory-bank size. Smoke tests and CI
must request the small fixture explicitly with `--entries 8`; 8 entries is not a
publishable performance baseline. Keep each size in a
separate directory so its bank, queries, and expected results stay together, for
example `python tools/generate_corpus.py --entries 100 --output bench_data/100`.
Run the harness once per directory and use a separate results directory for each
size.

For an extended batch sweep, generate a query count divisible by every batch
size and pass the matching list to the harness, for example:

```sh
python tools/generate_corpus.py --entries 100 --query-count 1024 \
  --output bench_data/100
python tools/bench_harness.py ... --batches 1,2,4,8,16,32,64,128,256,512,1024
```

`rag_bench` loads the bank and constructs the selected backend once, reports bank-load and cold-start records separately, performs discarded warmups, and emits raw CSV records for every measured query. The harness launches each backend and batch size in 10 independent trials, writes one raw CSV per trial, and reports pooled latency plus run-to-run throughput and startup variability. It refuses to publish performance rows for a backend that fails top-1, gate, or score-error checks against the FP32 reference. `render_report.py --run-dir results/YYYY-MM-DD` combines authoritative release data with CUDA-event hardware data into the bundle's `report/` directory. Run only the backends available on the benchmark host.

For the standard scaling campaign, `tools/run_bank_sweep.py` generates and runs the
deterministic 10/100/3,400/10,000-entry matrix. The separate 8-entry corpus remains only
for smoke tests and CI. For example:

```sh
python tools/run_bank_sweep.py \
  --rag-bench build-cuda/rag_bench \
  --data-root bench_data/sweep \
  --results-root results/YYYY-MM-DD/scaling \
  --backends cpu,cuda-optimized,cuda-optimized:bf16 \
  --batches 1,4,16,32,128 --trials 10
python tools/validate_campaign.py results/YYYY-MM-DD/scaling/entries-3400
```

The 3,400-entry full-payload bank is approximately 1.34 GiB and the 10,000-entry bank approximately 3.93 GiB, excluding runtime allocations. Metric ownership is versioned in `docs/metric_contracts.json`. Telemetry energy requires an identified `--telemetry-power-rail` and at least two samples inside the measured phase; otherwise `J/query` remains empty.

The binary format is little-endian: 8-byte `RAGBANK1` magic, four uint32 values (`count`, 2048, 101, 1024), `count × 2048` float32 normalized keys, `count` uint64 payload offsets, then payloads. Each payload contains a uint32 UTF-8 description length, description bytes, and `101 × 1024` float32 visual values. The builder writes `memory.bank.json` with dimensions, source hash, and supplied metadata. Include checkpoint IDs, data split, image preprocessing, and selected patch indices in that metadata.

## Current limits

The CUDA reference backend still copies all scores to the host intentionally, providing a comparison point for the optimized backend. The optimized backend returns only the winning result, but still normalizes queries on the CPU. The bank stores visual features as float32, so its size may differ from the paper's reported 2.7 GB. The module does not yet generate TinyLlama or CLIP embeddings; those adapters require the team's agreed checkpoints and preprocessing. TT-NN/TT-Metal backends and hardware power samplers remain target-hardware integration work; the benchmark CLI and CSV formats are ready to accept them.

## Performance evaluation

The planned Tenstorrent p150a and NVIDIA Jetson comparison uses three evaluation layers:
the Application Layer, Hardware Layer, and Stage Layer. During the first semester, each
of YOLOv8, CLIP ViT-L, RAG, and TinyLlama is treated as the application under test. During
the second semester, the integrated autonomous-vehicle VLM becomes the application under
test. See [docs/performance_evaluation.md](docs/performance_evaluation.md) for the project goal,
standards basis, cross-subsystem metrics, subsystem stages, RAG-specific plan, fairness
rules, and result-table templates.

## Project

ECEN Capstone: Tenstorrent Blackhole p150a Benchmarker. Project members: Annie Li, Paul Rodriguez, Rian Hickey, and Travis Mora.
