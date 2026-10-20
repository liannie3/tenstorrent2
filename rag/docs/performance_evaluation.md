# RAG Subsystem Benchmark Report

This report contains quality-gated measurements for the standalone RAG
subsystem. See the raw CSV files for full precision. 

## Run configuration

| Field | Value |
| --- | --- |
| Generated (UTC) | 2026-10-13T15:17:42.085116+00:00 |
| Host | Windows-10-10.0.26200-SP0 |
| Python | 3.10.12 |
| Backends | cpu |
| Batch sizes | 1, 2, 4, 8, 16, 32, 64, 128 |
| Query corpus | 128 queries |
| Memory bank | 8 entries |
| Similarity threshold | 0.700 |
| Independent trials | 10 |
| Warmup iterations | 10 |
| Measured corpus passes/trial | 100 |

## Quality gate

All results must first pass a quality gate against the CPU reference

| Backend | Precision | Queries | Top-1 agreement (%) | Gate agreement (%) | Mean score error | Max score error | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU reference | FP32 | 128 | 100.00 | 100.00 | 2.44e-10 | 4.94e-10 | PASS |

## Application performance

Latency is measured from submission of a prepared query embedding until the complete
response is available to the caller.

| Backend | Precision | Batch | Trials | Requests | Request | P50 (ms) | P95 (ms) | P99 (ms) | Throughput mean (query/s) | Throughput CV (%) | Energy (J/query) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU reference | FP32 | 1 | 10 | 128,000 | Single request · all | 0.232 | 0.602 | 0.933 | 3,366 | 9.72 | — |
| CPU reference | FP32 | 1 | 10 | 127,000 | Single request · accepted | 0.233 | 0.604 | 0.934 | — | — | — |
| CPU reference | FP32 | 1 | 10 | 1,000 | Single request · rejected | 0.012 | 0.034 | 0.078 | — | — | — |
| CPU reference | FP32 | 2 | 10 | 64,000 | Batch request · all | 0.511 | 1.361 | 1.905 | 3,029 | 9.21 | — |
| CPU reference | FP32 | 4 | 10 | 32,000 | Batch request · all | 1.244 | 2.479 | 3.611 | 2,913 | 14.84 | — |
| CPU reference | FP32 | 8 | 10 | 16,000 | Batch request · all | 3.037 | 5.116 | 6.796 | 2,446 | 17.23 | — |
| CPU reference | FP32 | 16 | 10 | 8,000 | Batch request · all | 6.512 | 11.809 | 14.508 | 2,164 | 3.61 | — |
| CPU reference | FP32 | 32 | 10 | 4,000 | Batch request · all | 13.924 | 23.574 | 28.257 | 2,007 | 2.85 | — |
| CPU reference | FP32 | 64 | 10 | 2,000 | Batch request · all | 32.902 | 46.225 | 53.567 | 1,797 | 8.29 | — |
| CPU reference | FP32 | 128 | 10 | 1,000 | Batch request · all | 68.243 | 85.190 | 94.481 | 1,777 | 5.05 | — |

## Startup

Values are the mean ± sample standard deviation across independent trials.

| Backend | Precision | Batch | Bank load (ms) | Backend init (ms) | First request (ms) | Cold start (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU reference | FP32 | 1 | 0.222 ± 0.083 | 0.004 ± 0.001 | 0.422 ± 0.090 | 1.594 ± 0.163 |
| CPU reference | FP32 | 2 | 0.249 ± 0.070 | 0.004 ± 0.002 | 0.475 ± 0.127 | 1.789 ± 0.387 |
| CPU reference | FP32 | 4 | 0.271 ± 0.085 | 0.005 ± 0.002 | 0.476 ± 0.110 | 1.975 ± 0.437 |
| CPU reference | FP32 | 8 | 0.277 ± 0.072 | 0.004 ± 0.001 | 0.483 ± 0.114 | 1.906 ± 0.400 |
| CPU reference | FP32 | 16 | 0.212 ± 0.041 | 0.004 ± 0.001 | 0.428 ± 0.150 | 1.627 ± 0.237 |
| CPU reference | FP32 | 32 | 0.254 ± 0.093 | 0.005 ± 0.002 | 0.439 ± 0.115 | 1.733 ± 0.293 |
| CPU reference | FP32 | 64 | 0.272 ± 0.104 | 0.005 ± 0.002 | 0.460 ± 0.118 | 1.815 ± 0.305 |
| CPU reference | FP32 | 128 | 0.242 ± 0.061 | 0.006 ± 0.003 | 0.525 ± 0.150 | 2.046 ± 0.401 |

## RAG stages

Payload retrieval applies only to accepted queries.

| Backend | Precision | Batch | Path | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Critical path (%) |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| CPU reference | FP32 | 1 | all | Vector retrieval | 0.012 | 0.026 | 0.048 | 4.9 |
| CPU reference | FP32 | 1 | accepted | Vector retrieval | 0.012 | 0.026 | 0.048 | 4.8 |
| CPU reference | FP32 | 1 | rejected | Vector retrieval | 0.012 | 0.033 | 0.075 | 99.0 |
| CPU reference | FP32 | 1 | accepted | Payload retrieval | 0.221 | 0.577 | 0.890 | 95.2 |
| CPU reference | FP32 | 2 | all | Vector retrieval | 0.024 | 0.066 | 0.098 | 4.8 |
| CPU reference | FP32 | 4 | all | Vector retrieval | 0.048 | 0.113 | 0.192 | 4.5 |
| CPU reference | FP32 | 8 | all | Vector retrieval | 0.101 | 0.249 | 0.382 | 4.1 |
| CPU reference | FP32 | 16 | all | Vector retrieval | 0.283 | 0.631 | 0.780 | 4.1 |
| CPU reference | FP32 | 32 | all | Vector retrieval | 0.564 | 1.237 | 1.539 | 3.9 |
| CPU reference | FP32 | 64 | all | Vector retrieval | 1.136 | 1.936 | 2.997 | 3.5 |
| CPU reference | FP32 | 128 | all | Vector retrieval | 2.313 | 3.572 | 4.806 | 3.5 |

## Storage footprint

| Backend | Precision | Entries | Bank file | Key data | Offset table | Mean payload |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU reference | FP32 | 8 | 3.22 MiB | 64.00 KiB | 64 B | 404.02 KiB |

## Runtime memory

| Backend | Precision | Batch | Peak host RSS | Persistent device | Peak device | Workspace |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU reference | FP32 | 1 | 178.36 MiB | — | — | — |
| CPU reference | FP32 | 2 | 178.72 MiB | — | — | — |
| CPU reference | FP32 | 4 | 179.52 MiB | — | — | — |
| CPU reference | FP32 | 8 | 181.13 MiB | — | — | — |
| CPU reference | FP32 | 16 | 184.39 MiB | — | — | — |
| CPU reference | FP32 | 32 | 190.90 MiB | — | — | — |
| CPU reference | FP32 | 64 | 203.91 MiB | — | — | — |
| CPU reference | FP32 | 128 | 229.47 MiB | — | — | — |

## Hardware

CUDA hardware rows come from CUDA events recorded around operations in the same stream as
the optimized search. Device time is the median total CUDA duration divided by batch size.
Transfer time is the median sum of H2D and D2H durations; transfer bytes are normalized per
query. Effective GOP/s uses `2 * entries * 2048 * batch` similarity operations, and
effective GB/s uses the resident key-matrix bytes divided by GEMM time. The raw hardware
CSV additionally retains batch P50/P95/P99 and individual stage medians. Blank values mean
that the backend or build did not provide device timing; they do not mean zero.

Application latency tables must come from an unprofiled release run. Hardware rows may
come from an event-timing build after its overhead has been compared with that release
baseline. Nsight Systems captures are diagnostic and are not a source of percentile data.

| Backend | Precision | Batch | Device time (µs/query) | Transfer time (µs) | Transfer (B/query) | Effective GOP/s | Effective GB/s | Bandwidth (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU reference | FP32 | 1 | — | — | — | — | — | — |
| CPU reference | FP32 | 2 | — | — | — | — | — | — |
| CPU reference | FP32 | 4 | — | — | — | — | — | — |
| CPU reference | FP32 | 8 | — | — | — | — | — | — |
| CPU reference | FP32 | 16 | — | — | — | — | — | — |
| CPU reference | FP32 | 32 | — | — | — | — | — | — |
| CPU reference | FP32 | 64 | — | — | — | — | — | — |
| CPU reference | FP32 | 128 | — | — | — | — | — | — |
