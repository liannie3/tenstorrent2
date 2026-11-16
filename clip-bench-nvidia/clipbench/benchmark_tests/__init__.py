# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            __init__.py
# CREATED:         10/25/2026
# REVISION:        1.0
# ======================================= #

from . import cpu_pure_latency
from . import deep_metrics
from . import gpu_pure_latency
from . import hardware_perf

## cpu_pure_latency_tt and gpu_pure_latency_tt are deliberately not imported
## here: they import ttnn, which would break run_nvidia.py on the Orin. run_tt.py
## imports them directly.

## The name is also the results subdirectory, so each test's CSVs keep the same
## stage names without colliding.
TESTS = {
    'cpu_pure_latency': cpu_pure_latency.run,
    'gpu_pure_latency': gpu_pure_latency.run,
    'hardware_perf': hardware_perf.run,
    'nvtx_stats': deep_metrics.run,
}


def available():
    return tuple(TESTS)
