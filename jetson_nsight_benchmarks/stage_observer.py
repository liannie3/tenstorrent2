"""Stage labels and measurements stored inside the native Nsight report."""
from contextlib import contextmanager
import time


class StageObserver:
    def __init__(self, mode, device, torch):
        self.mode, self.device, self.torch = mode, device, torch
        self.tokens = []
        self.input_tokens = None

    @contextmanager
    def span(self, name):
        # Stage fences are ONLY used in the latency experiment. Other tests
        # retain asynchronous launches and use native CUDA correlation.
        if self.mode == 'latency':
            self.torch.cuda.synchronize(self.device)
        with self.torch.cuda.nvtx.range(name):
            wall = time.perf_counter()
            cpu = time.process_time() if self.mode == 'cpu' else None
            yield
            if self.mode == 'latency':
                self.torch.cuda.synchronize(self.device)
            wall_ms = (time.perf_counter()-wall)*1000
            if self.mode == 'latency':
                self.torch.cuda.nvtx.mark(f'metric stage={name} wall_ms={wall_ms:.6f}')
            elif self.mode == 'cpu':
                cpu_ms = (time.process_time()-cpu)*1000
                percent = 100*cpu_ms/wall_ms if wall_ms else 0
                self.torch.cuda.nvtx.mark(
                    f'metric stage={name} process_cpu_ms={cpu_ms:.6f} '
                    f'host_wall_ms={wall_ms:.6f} cpu_percent_one_core={percent:.3f}')

    @contextmanager
    def request(self, stage):
        start = time.perf_counter()
        with self.span('llm_total'):
            yield self
            n = len(self.tokens)
            ttft = (self.tokens[0]-start)*1000 if n else None
            tpot = (self.tokens[-1]-self.tokens[0])*1000/(n-1) if n>1 else None
            self.torch.cuda.nvtx.mark(f'llm_tokens input={self.input_tokens} output={n} '
                                      f'ttft_ms={ttft} tpot_ms={tpot}')

    def input_ready(self, count):
        self.input_tokens = int(count)

    def token_ready(self):
        # Called after the decoder's existing .item(), which brings the token to CPU.
        self.tokens.append(time.perf_counter())
