# RAG Retrieval Research Questions

## Purpose

This document turns possible extensions to the Jetson-versus-p150a retrieval benchmark
into testable research questions. The primary comparison remains quality-constrained:
performance improvements are valid only when retrieval and threshold decisions meet the
declared quality target.

## Priority questions

### 1. When does device residency help?

**Question:** Does device-resident retrieval improve a standalone request, or does its
main benefit appear only when both the embedding producer and payload consumer are also
on the accelerator?

Compare:

1. Host query → device retrieval → host payload.
2. Host query → device retrieval and payload lookup → host payload.
3. Device-resident query → device retrieval → host result.
4. Device-resident query → device retrieval → device-resident payload.

Measure P95 latency, PCIe bytes/query, device time, queries/s, J/query, and device memory.
TT-Metal command queues order upload, execution, and download commands asynchronously, so
also report how much transfer time is hidden by overlap [TT-Metal command queues][tt-cq].

### 2. When does fusion beat composed operations?

**Question:** At what bank size and batch size does fused streaming search outperform a
composed matmul followed by reduction?

Compare:

- Matmul → materialized score matrix → top-1 kernel.
- Tiled similarity → running top-1 → threshold gate.
- A partially fused implementation.

Report P95 latency, queries/s, score-workspace bytes, memory traffic, device time, and
quality agreement. Faiss reports that fusing distance calculation and selection avoided
an extra score-matrix pass and improved its exact-search implementation; this motivates
the experiment but does not predict the p150a result [Faiss GPU search][faiss].

### 3. What precision gives the best quality–memory–performance frontier?

**Question:** Which key/query format minimizes memory traffic and latency without changing
retrieval or threshold decisions?

Test FP32 reference, controlled BF16, FP16 where comparable, BLOCKFP8, and useful mixed-
precision combinations. Report top-1 agreement, gate agreement, score error, P95 latency,
queries/s, bank bytes, effective bandwidth, and J/query. Include queries immediately
around the `0.7` threshold. Blackhole Tensix cores support BF16, FP16, FP8, and block
floating-point formats [Blackhole specifications][blackhole].

### 4. Where are the dispatch, bandwidth, and compute regimes?

**Question:** How do the limiting resources change with bank size, batch size, core count,
and precision?

Measure host submission time, device active time, operations/s, effective key bandwidth,
percentage of sustainable bandwidth, P95 latency, and core utilization. Identify rather
than assume the crossovers among:

- Small-workload dispatch/synchronization limits.
- Batch-1 memory-bandwidth limits.
- Larger-batch compute utilization and query reuse.

### 5. How should the bank be distributed across Tensix cores?

**Question:** Which sharding and reduction scheme gives the best scaling without creating
load imbalance or excessive global reduction cost?

Compare row partitioning, replicated-query/sharded-bank layouts, DRAM-channel mappings,
and one-level versus hierarchical top-1 reduction. Report throughput scaling efficiency,
slowest-core time, reduction time, DRAM bandwidth, and NoC traffic. TT-Metal explicitly
streams tiles from device DRAM into per-core L1 circular buffers, making layout and
channel use first-class design choices [TT-Metal matmul lab][tt-matmul].

### 6. Should keys and payloads use different residency policies?

**Question:** Does the best location for the small searchable keys differ from the best
location for the much larger payloads?

Compare keys-only device residency, device visual features with host descriptions,
complete device residency, and an optional hot-payload cache. Report retrieval latency,
payload latency, PCIe bytes, host/device memory, cold/warm behavior, maximum entries, and
J/query. Treat payload compression as a separate quality-controlled experiment.

## Extension questions

### 7. Where is the exact-to-approximate search crossover?

Increase bank size until exhaustive search misses its latency or capacity target, then
compare ANN search at matched recall. Report the break-even bank size, P95 latency,
queries/s, Recall@k, index bytes/vector, build time, and peak build memory. CAGRA shows
that GPU graph ANN can serve both small- and large-batch searches efficiently, but its
results must be compared at controlled recall [CAGRA][cagra].

Implement this only after exact search is stable; an ANN backend can become a separate
project.

### 8. Can transfers be hidden rather than eliminated?

Compare blocking execution with double-buffered queries, query `n+1` transfer overlapped
with query `n` computation, and payload transfer overlapped with the next retrieval.
Report raw transfer time, critical-path transfer time, achieved overlap, and queue depth.
TeleRAG demonstrates that data movement and computation can overlap, so bytes transferred
and latency added by transfer are distinct measurements [TeleRAG][telerag].

### 9. How robust is the relevance gate across platforms and precisions?

Construct queries with reference scores well below, immediately below, near, immediately
above, and well above `0.7`, plus tied and near-tied candidates. Report score error,
top-1 agreement, gate disagreement, false accepts, false rejects, and deterministic
tie-breaking. This is project-specific: a small numerical error can change whether a
payload is returned even when average score error is low.

### 10. What does bank mutability cost?

If the production bank must change during service, compare full rebuild/upload,
append-only batches, in-place replacement, tombstones, and double-buffered bank swaps.
Measure update latency and throughput, temporary memory, query P99 impact, and consistency.
Recent CPU–GPU vector-search work identifies a tradeoff between GPU query throughput,
limited device capacity, and efficient online updates [SVFusion][svfusion].

## Recommended scope

Complete Questions 1–6 first. They directly test p150a architectural hypotheses using the
existing exact-retrieval workload. Questions 7–10 are extensions and should not delay the
controlled CUDA-versus-Tenstorrent comparison.

The unifying research question is:

> Under what bank size, batch size, precision, data placement, and pipeline-residency
> conditions does a p150a dataflow implementation outperform an equivalent CUDA
> implementation in latency, throughput, memory efficiency, and energy without changing
> retrieval decisions?

[blackhole]: https://docs.tenstorrent.com/aibs/blackhole/index.html
[cagra]: https://arxiv.org/abs/2308.15136
[faiss]: https://arxiv.org/abs/1702.08734
[svfusion]: https://www.vldb.org/pvldb/vol19/p1074-yang.pdf
[telerag]: https://arxiv.org/abs/2502.20969
[tt-cq]: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/dram_loopback.html
[tt-matmul]: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html
