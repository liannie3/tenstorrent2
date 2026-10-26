# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            __init__.py
# CREATED:         10/25/2026
# REVISION:        1.0
# ======================================= #

from . import deep_metrics
from . import pure_latency

## run_nvidia.py's --test choices. Keeping the mapping here means the CLI and
## the modules cannot drift apart.
##
## hardware_perf is deliberately NOT in here -- it is a wrapper, not a test, so
## run_nvidia.py reaches for it separately.
TESTS = {
    'latency': pure_latency.run,
    'nvtx': deep_metrics.run,
}


def available():
    return tuple(TESTS)
