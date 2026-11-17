flowchart TD
    1[1. Environment & Dataset Pre-Check] --> 2{Target Hardware Testbed}

    subgraph NVIDIA [NVIDIA Jetson AGX Orin Branch]
        2 -->|Unified Memory SoC| 3A[2a. Launch Nsight / nvpmodel Telemetry]
        3A --> 4A[3a. Execute Warmup + 5x Runs CUDA / TensorRT]
    end

    subgraph Tenstorrent [Tenstorrent Blackhole p150a Branch]
        2 -->|Discrete PCIe Architecture| 3B[2b. Launch tt-smi Telemetry Daemon]
        3B --> 4B[3b. Execute Warmup + 5x Runs TT-Metalium / TT-NN]
    end

    4A --> 5[4. Aggregate Telemetry & Per-Stage Latencies]
    4B --> 5

    5 --> 6[5. Validate Statistical Repeatability σ ≤ 5%]
    6 --> 7[6. Export Standardized JSON/CSV Report]