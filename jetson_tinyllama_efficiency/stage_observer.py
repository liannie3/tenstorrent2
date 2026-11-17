"""Connect the separate latency/CPU/GPU scripts to the original NsightBuffer.

The uploaded benchmark_stages.py imports StageObserver; this bridge preserves the
same TinyLlamaStage.answer(benchmark=...) interface rather than adding a second
or simulated decoder. Each measured request gets a fresh observer instance.
"""
from nsight_buffer import NsightBuffer


class StageObserver(NsightBuffer):
    """All three experiments share application timings and NVTX stage ranges.

    The experiment type selects Nsight's native collection options in
    benchmark_stages.profile_options(); it is NOT a different language model.
    """

    def __init__(self, mode, device, torch, *, interval_ms=100,
                 tegrastats=False, tegrastats_interval_ms=100):
        # ==================== EXPERIMENT CATEGORY ====================
        if mode not in ('latency', 'cpu', 'gpu'):
            raise ValueError(f'Unknown benchmark category: {mode}')
        self.mode = mode
        self.device = device
        self.torch = torch

        # ==================== REUSE ORIGINAL HOOKS, TIMERS AND MONITORS ====================
        super().__init__(interval_ms,
                         tegrastats=tegrastats,
                         tegrastats_interval_ms=tegrastats_interval_ms)
