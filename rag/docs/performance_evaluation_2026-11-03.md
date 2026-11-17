# RAG Subsystem Benchmark Report — 2026-11-03

This report contains quality-gated measurements for the November 3 NVIDIA benchmark run. See the raw CSV files for full precision.

The measured memory bank contains only 8 entries and is therefore a smoke fixture, not the 3,400-entry paper-comparison baseline.

## Run configuration

| Field | Value |
| --- | --- |
| Generated (UTC) | 2026-11-03T16:14:58.187794+00:00 |
| Host | Linux-5.15.148-tegra-aarch64-with-glibc2.35 |
| Python | 3.10.12 |
| Backends | cpu, cuda-reference, cuda-optimized, cuda-optimized:bf16 |
| Batch sizes | 1, 4, 16, 128 |
| Query corpus | 128 queries |
| Memory bank | 8 entries |
| Similarity threshold | 0.700 |
| Independent trials | 3 |
| Warmup iterations | 10 |
| Measured corpus passes/trial | 100 |

## Quality gate

| Backend | Precision | Queries | Top-1 agreement (%) | Gate agreement (%) | Mean score error | Max score error | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 128 | 100.000000 | 100.000000 | 2.44327109e-10 | 4.94384755e-10 | PASS |
| CUDA reference | FP32 | 128 | 100.000000 | 100.000000 | 1.73840439e-08 | 6.00430603e-08 | PASS |
| CUDA optimized | FP32 | 128 | 100.000000 | 100.000000 | 2.57899068e-08 | 1.19682373e-07 | PASS |
| CUDA optimized | BF16 | 128 | 100.000000 | 100.000000 | 6.2330562e-05 | 0.000289380994 | PASS |

## Application performance

| Backend | Precision | Batch | Trials | Requests | Request | P50 (ms) | P95 (ms) | P99 (ms) | Throughput mean (query/s) | Throughput CV (%) | Energy (J/query) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 1 | 3 | 38400 | Single request · all | 0.095552 | 0.118431 | 0.139231 | 10098.709 | 3.373 | — |
| CPU reference | FP32 | 1 | 3 | 38100 | Single request · accepted | 0.095680 | 0.118527 | 0.139264 | — | — | — |
| CPU reference | FP32 | 1 | 3 | 300 | Single request · rejected | 0.018383 | 0.018464 | 0.019007 | — | — | — |
| CPU reference | FP32 | 4 | 3 | 9600 | Batch request · all | 0.821932 | 0.861147 | 0.893407 | 4355.512 | 0.812 | — |
| CPU reference | FP32 | 16 | 3 | 2400 | Batch request · all | 3.836874 | 4.018268 | 4.241776 | 3730.551 | 1.652 | — |
| CPU reference | FP32 | 128 | 3 | 300 | Batch request · all | 32.445529 | 32.973162 | 33.248456 | 3329.710 | 0.494 | — |
| CUDA reference | FP32 | 1 | 3 | 38400 | Single request · all | 0.122975 | 0.153887 | 0.173152 | 7809.702 | 6.222 | — |
| CUDA reference | FP32 | 1 | 3 | 38100 | Single request · accepted | 0.123072 | 0.153921 | 0.173570 | — | — | — |
| CUDA reference | FP32 | 1 | 3 | 300 | Single request · rejected | 0.034048 | 0.041389 | 0.079876 | — | — | — |
| CUDA reference | FP32 | 4 | 3 | 9600 | Batch request · all | 0.935993 | 0.992899 | 2.990948 | 3750.796 | 6.041 | — |
| CUDA reference | FP32 | 16 | 3 | 2400 | Batch request · all | 4.248561 | 5.101901 | 5.746217 | 3379.972 | 2.948 | — |
| CUDA reference | FP32 | 128 | 3 | 300 | Batch request · all | 35.566081 | 36.016210 | 36.367136 | 3079.079 | 0.462 | — |
| CUDA optimized | FP32 | 1 | 3 | 38400 | Single request · all | 0.154623 | 0.181761 | 0.195006 | 6302.235 | 4.399 | — |
| CUDA optimized | FP32 | 1 | 3 | 38100 | Single request · accepted | 0.154718 | 0.181791 | 0.195102 | — | — | — |
| CUDA optimized | FP32 | 1 | 3 | 300 | Single request · rejected | 0.057391 | 0.062681 | 0.070852 | — | — | — |
| CUDA optimized | FP32 | 4 | 3 | 9600 | Batch request · all | 0.865816 | 0.900633 | 0.925337 | 4159.092 | 0.501 | — |
| CUDA optimized | FP32 | 16 | 3 | 2400 | Batch request · all | 3.791757 | 3.851893 | 3.902157 | 3811.259 | 0.339 | — |
| CUDA optimized | FP32 | 128 | 3 | 300 | Batch request · all | 31.323181 | 31.677632 | 31.778875 | 3418.600 | 0.316 | — |
| CUDA optimized | BF16 | 1 | 3 | 38400 | Single request · all | 0.164158 | 0.191614 | 0.298180 | 5698.274 | 5.934 | — |
| CUDA optimized | BF16 | 1 | 3 | 38100 | Single request · accepted | 0.164319 | 0.191678 | 0.299169 | — | — | — |
| CUDA optimized | BF16 | 1 | 3 | 300 | Single request · rejected | 0.063168 | 0.070952 | 0.075105 | — | — | — |
| CUDA optimized | BF16 | 4 | 3 | 9600 | Batch request · all | 0.891383 | 1.495959 | 3.102310 | 3688.611 | 3.663 | — |
| CUDA optimized | BF16 | 16 | 3 | 2400 | Batch request · all | 3.844359 | 4.119706 | 5.239390 | 3705.141 | 1.205 | — |
| CUDA optimized | BF16 | 128 | 3 | 300 | Batch request · all | 30.547027 | 31.647039 | 33.075871 | 3575.637 | 0.450 | — |

## Startup

| Backend | Precision | Batch | Bank load (ms) | Backend init (ms) | First request (ms) | Cold start (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 1 | 0.148298 ± 0.003511 | 0.000736 ± 0.000279 | 0.370814 ± 0.030235 | 1.275012 ± 0.074307 |
| CPU reference | FP32 | 4 | 0.144949 ± 0.007263 | 0.000725 ± 0.000176 | 0.359027 ± 0.010799 | 1.249263 ± 0.045818 |
| CPU reference | FP32 | 16 | 0.152116 ± 0.008656 | 0.005675 ± 0.008554 | 0.392948 ± 0.086727 | 1.334702 ± 0.224075 |
| CPU reference | FP32 | 128 | 0.145610 ± 0.005726 | 0.000384 ± 0.000085 | 0.336563 ± 0.010586 | 1.187470 ± 0.008656 |
| CUDA reference | FP32 | 1 | 0.169567 ± 0.044528 | 49.827204 ± 1.035457 | 0.702374 ± 0.171861 | 51.495630 ± 1.158995 |
| CUDA reference | FP32 | 4 | 0.148714 ± 0.012233 | 48.269886 ± 0.461780 | 0.611068 ± 0.008725 | 49.747487 ± 0.477563 |
| CUDA reference | FP32 | 16 | 0.143071 ± 0.001857 | 48.111902 ± 0.205844 | 0.613233 ± 0.014369 | 49.607603 ± 0.217312 |
| CUDA reference | FP32 | 128 | 0.193182 ± 0.063609 | 52.822349 ± 4.766621 | 0.831641 ± 0.201993 | 54.815069 ± 4.844217 |
| CUDA optimized | FP32 | 1 | 0.140948 ± 0.003624 | 61.912607 ± 1.589170 | 38.728201 ± 0.167292 | 101.500588 ± 1.744218 |
| CUDA optimized | FP32 | 4 | 0.142761 ± 0.001887 | 60.683616 ± 0.261027 | 38.607838 ± 0.185189 | 100.145985 ± 0.386746 |
| CUDA optimized | FP32 | 16 | 0.142751 ± 0.007354 | 60.795026 ± 0.288357 | 38.886892 ± 0.257875 | 100.545932 ± 0.051074 |
| CUDA optimized | FP32 | 128 | 0.148244 ± 0.003622 | 62.997594 ± 1.442817 | 38.825832 ± 0.143413 | 102.702832 ± 1.466487 |
| CUDA optimized | BF16 | 1 | 0.145193 ± 0.002673 | 61.716516 ± 1.862621 | 49.602320 ± 0.521918 | 112.186827 ± 2.349524 |
| CUDA optimized | BF16 | 4 | 0.148724 ± 0.006754 | 62.244234 ± 1.082145 | 49.666058 ± 0.650189 | 112.786976 ± 1.724698 |
| CUDA optimized | BF16 | 16 | 0.148297 ± 0.008588 | 61.214675 ± 0.732973 | 49.584887 ± 1.058067 | 111.695755 ± 1.426916 |
| CUDA optimized | BF16 | 128 | 0.161556 ± 0.015971 | 65.240368 ± 3.355444 | 49.271731 ± 0.434682 | 115.401722 ± 3.787540 |

## RAG stages

| Backend | Precision | Batch | Path | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Critical path (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 1 | all | Vector retrieval | 0.018304 | 0.018464 | 0.019135 | 18.998 |
| CPU reference | FP32 | 1 | accepted | Vector retrieval | 0.018304 | 0.018464 | 0.019136 | 18.878 |
| CPU reference | FP32 | 1 | rejected | Vector retrieval | 0.018304 | 0.018368 | 0.018911 | 99.551 |
| CPU reference | FP32 | 1 | accepted | Payload retrieval | 0.077312 | 0.099776 | 0.119553 | 81.122 |
| CPU reference | FP32 | 4 | all | Vector retrieval | 0.072863 | 0.073823 | 0.084321 | 8.948 |
| CPU reference | FP32 | 16 | all | Vector retrieval | 0.297918 | 0.312288 | 0.350372 | 7.801 |
| CPU reference | FP32 | 128 | all | Vector retrieval | 2.328434 | 2.356767 | 2.379672 | 7.184 |
| CUDA reference | FP32 | 1 | all | Vector retrieval | 0.033759 | 0.040128 | 0.048416 | 27.485 |
| CUDA reference | FP32 | 1 | accepted | Vector retrieval | 0.033759 | 0.040128 | 0.048224 | 27.326 |
| CUDA reference | FP32 | 1 | rejected | Vector retrieval | 0.033920 | 0.041231 | 0.079525 | 99.625 |
| CUDA reference | FP32 | 1 | accepted | Payload retrieval | 0.088895 | 0.119039 | 0.129055 | 72.674 |
| CUDA reference | FP32 | 4 | all | Vector retrieval | 0.172990 | 0.188417 | 2.221429 | 21.492 |
| CUDA reference | FP32 | 16 | all | Vector retrieval | 0.706843 | 1.535466 | 2.068855 | 18.095 |
| CUDA reference | FP32 | 128 | all | Vector retrieval | 5.252087 | 5.334604 | 5.463777 | 14.786 |
| CUDA optimized | FP32 | 1 | all | Vector retrieval | 0.056672 | 0.062783 | 0.071807 | 36.784 |
| CUDA optimized | FP32 | 1 | accepted | Vector retrieval | 0.056672 | 0.062783 | 0.071807 | 36.600 |
| CUDA optimized | FP32 | 1 | rejected | Vector retrieval | 0.057248 | 0.062492 | 0.070692 | 99.758 |
| CUDA optimized | FP32 | 1 | accepted | Payload retrieval | 0.097472 | 0.123873 | 0.131263 | 63.400 |
| CUDA optimized | FP32 | 4 | all | Vector retrieval | 0.101247 | 0.106207 | 0.117535 | 11.786 |
| CUDA optimized | FP32 | 16 | all | Vector retrieval | 0.259005 | 0.279463 | 0.314156 | 6.946 |
| CUDA optimized | FP32 | 128 | all | Vector retrieval | 1.474690 | 1.516657 | 1.565981 | 4.706 |
| CUDA optimized | BF16 | 1 | all | Vector retrieval | 0.062880 | 0.072098 | 0.202591 | 41.072 |
| CUDA optimized | BF16 | 1 | accepted | Vector retrieval | 0.062880 | 0.072127 | 0.202884 | 40.902 |
| CUDA optimized | BF16 | 1 | rejected | Vector retrieval | 0.063007 | 0.070824 | 0.074946 | 99.776 |
| CUDA optimized | BF16 | 1 | accepted | Payload retrieval | 0.100223 | 0.125119 | 0.135166 | 59.098 |
| CUDA optimized | BF16 | 4 | all | Vector retrieval | 0.116383 | 0.701248 | 2.298863 | 21.380 |
| CUDA optimized | BF16 | 16 | all | Vector retrieval | 0.295341 | 0.425827 | 1.611896 | 8.897 |
| CUDA optimized | BF16 | 128 | all | Vector retrieval | 1.414352 | 2.005522 | 3.081000 | 4.705 |

## Storage footprint

| Backend | Precision | Entries | Bank file | Key data | Offset table | Mean payload |
| --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 8 | 3.22 MiB | 64.00 KiB | 64 B | 404.02 KiB |
| CUDA reference | FP32 | 8 | 3.22 MiB | 64.00 KiB | 64 B | 404.02 KiB |
| CUDA optimized | FP32 | 8 | 3.22 MiB | 64.00 KiB | 64 B | 404.02 KiB |
| CUDA optimized | BF16 | 8 | 3.22 MiB | 64.00 KiB | 64 B | 404.02 KiB |

## Runtime memory

| Backend | Precision | Batch | Peak host RSS | Persistent device | Peak device | Workspace |
| --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 1 | 122.11 MiB | — | — | — |
| CPU reference | FP32 | 4 | 123.25 MiB | — | — | — |
| CPU reference | FP32 | 16 | 128.10 MiB | — | — | — |
| CPU reference | FP32 | 128 | 172.69 MiB | — | — | — |
| CUDA reference | FP32 | 1 | 221.99 MiB | — | — | — |
| CUDA reference | FP32 | 4 | 223.06 MiB | — | — | — |
| CUDA reference | FP32 | 16 | 228.05 MiB | — | — | — |
| CUDA reference | FP32 | 128 | 272.74 MiB | — | — | — |
| CUDA optimized | FP32 | 1 | 298.43 MiB | — | — | — |
| CUDA optimized | FP32 | 4 | 304.43 MiB | — | — | — |
| CUDA optimized | FP32 | 16 | 309.52 MiB | — | — | — |
| CUDA optimized | FP32 | 128 | 362.76 MiB | — | — | — |
| CUDA optimized | BF16 | 1 | 328.35 MiB | — | — | — |
| CUDA optimized | BF16 | 4 | 336.76 MiB | — | — | — |
| CUDA optimized | BF16 | 16 | 341.83 MiB | — | — | — |
| CUDA optimized | BF16 | 128 | 388.14 MiB | — | — | — |

## Hardware

| Backend | Precision | Batch | Device time (µs/query) | Transfer time (µs) | Transfer (B/query) | Effective GOP/s | Effective GB/s | Bandwidth (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 1 | — | — | — | — | — | — |
| CPU reference | FP32 | 4 | — | — | — | — | — | — |
| CPU reference | FP32 | 16 | — | — | — | — | — | — |
| CPU reference | FP32 | 128 | — | — | — | — | — | — |
| CUDA reference | FP32 | 1 | — | — | — | — | — | — |
| CUDA reference | FP32 | 4 | — | — | — | — | — | — |
| CUDA reference | FP32 | 16 | — | — | — | — | — | — |
| CUDA reference | FP32 | 128 | — | — | — | — | — | — |
| CUDA optimized | FP32 | 1 | — | — | — | — | — | — |
| CUDA optimized | FP32 | 4 | — | — | — | — | — | — |
| CUDA optimized | FP32 | 16 | — | — | — | — | — | — |
| CUDA optimized | FP32 | 128 | — | — | — | — | — | — |
| CUDA optimized | BF16 | 1 | — | — | — | — | — | — |
| CUDA optimized | BF16 | 4 | — | — | — | — | — | — |
| CUDA optimized | BF16 | 16 | — | — | — | — | — | — |
| CUDA optimized | BF16 | 128 | — | — | — | — | — | — |

## Thermal telemetry

| Backend | Precision | Batch | Provider | Samples/errors | Initial temperature (°C) | Mean temperature (°C) | Maximum temperature (°C) | Thermal rise (°C) | Throttled samples | Mean power (W) | Mean core clock (MHz) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU reference | FP32 | 1 | jetson | 6/0 | 53.625 | 53.760 | 53.875 | 0.062 | 0 | 5.451 | — |
| CPU reference | FP32 | 4 | jetson | 12/0 | 54.093 | 53.919 | 54.187 | -0.187 | 0 | 5.389 | — |
| CPU reference | FP32 | 16 | jetson | 15/0 | 53.718 | 53.964 | 54.250 | 0.375 | 0 | 5.527 | — |
| CPU reference | FP32 | 128 | jetson | 18/0 | 54.281 | 54.012 | 54.375 | -0.438 | 0 | 5.389 | — |
| CUDA reference | FP32 | 1 | jetson | 9/0 | 53.875 | 53.965 | 54.093 | 0.218 | 0 | 5.664 | — |
| CUDA reference | FP32 | 4 | jetson | 16/0 | 54.062 | 53.996 | 54.375 | -0.094 | 0 | 5.682 | — |
| CUDA reference | FP32 | 16 | jetson | 18/0 | 54.156 | 54.043 | 54.218 | -0.188 | 0 | 5.643 | — |
| CUDA reference | FP32 | 128 | jetson | 18/0 | 54.031 | 54.099 | 54.312 | 0.219 | 0 | 5.347 | — |
| CUDA optimized | FP32 | 1 | jetson | 11/0 | 54.218 | 54.096 | 54.281 | -0.156 | 0 | 5.637 | — |
| CUDA optimized | FP32 | 4 | jetson | 15/0 | 54.093 | 54.089 | 54.281 | -0.062 | 0 | 5.630 | — |
| CUDA optimized | FP32 | 16 | jetson | 15/0 | 54.281 | 54.110 | 54.343 | -0.188 | 0 | 5.707 | — |
| CUDA optimized | FP32 | 128 | jetson | 18/0 | 53.968 | 54.062 | 54.281 | 0.032 | 0 | 5.326 | — |
| CUDA optimized | BF16 | 1 | jetson | 12/0 | 54.218 | 54.153 | 54.312 | -0.093 | 0 | 5.642 | — |
| CUDA optimized | BF16 | 4 | jetson | 16/0 | 54.250 | 54.175 | 54.343 | -0.157 | 0 | 5.657 | — |
| CUDA optimized | BF16 | 16 | jetson | 15/0 | 54.187 | 54.241 | 54.531 | -0.062 | 0 | 5.680 | — |
| CUDA optimized | BF16 | 128 | jetson | 16/0 | 54.156 | 54.318 | 54.468 | 0.281 | 0 | 5.633 | — |
