"""Observe inference and write measurements only as NVTX events in Nsight.

CPU/RAM sampling uses Linux procfs on a background thread; it adds overhead.
WSL system readings describe the Linux VM. Missing readings are labeled.
Application timings are instrumented, not a profiler-free performance baseline.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import threading
import time


class NsightBuffer:
    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000
        self.handles = []
        self.tokens = []
        self.forward_count = 0
        self.input_count = None
        self.used = False
        self.stop = threading.Event()
        self.monitor_active = False

    @contextmanager
    def span(self, name):
        # Host-side NVTX range; GPU kernels appear on their own timeline tracks.
        import torch
        with torch.cuda.nvtx.range(name):
            yield

    @contextmanager
    def pipeline_scope(self, stage, name):
        """Monitor the whole pipeline; LLM hooks are installed only in answer()."""
        import torch
        thread = threading.Thread(target=self.monitor, args=(torch,), name='CPU_RAM_monitor', daemon=True)
        self.monitor_active = True
        thread.start()
        try:
            with self.span(name):
                torch.cuda.synchronize(stage.device)
                start = time.perf_counter()
                yield
                torch.cuda.synchronize(stage.device)
                torch.cuda.nvtx.mark(f'pipeline_latency_ms={(time.perf_counter()-start)*1000:.3f}')
        finally:
            self.stop.set()
            thread.join(timeout=2)
            self.monitor_active = False

    def input_ready(self, count):
        self.input_count = int(count)

    def token_ready(self):
        # Called after the existing token .item(); no extra GPU sync/copy.
        self.tokens.append(time.perf_counter())

    def monitor(self, torch):
        previous = None
        while not self.stop.is_set():
            now = time.perf_counter()
            values = []
            current = {}
            try:
                # Process CPU counters
                fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
                current['cpu_s'] = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
                # Process resident RAM
                rss = int(Path(f'/proc/{os.getpid()}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                values += [f'process_rss_bytes={rss}', f'threads={fields[17]}']
                if previous and 'cpu_s' in previous:
                    values.append(f'process_cpu_percent_one_core={100 * (current["cpu_s"] - previous["cpu_s"]) / (now - previous["time"]):.2f}')
            except (OSError, ValueError, IndexError):
                values.append('process_cpu_ram=unavailable')
            try:
                #System CPU Counters
                cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
                current.update(total=sum(cpu), idle=cpu[3] + cpu[4])
                if previous and 'total' in previous:
                    total = current['total'] - previous['total']
                    if total > 0:
                        values.append(f'system_cpu_percent={100 * (1 - (current["idle"] - previous["idle"]) / total):.2f}')
                #System RAM Information
                mem = {line.split(':')[0]: int(line.split()[1]) * 1024
                       for line in Path('/proc/meminfo').read_text().splitlines()
                       if len(line.split()) >= 3 and line.split()[2] == 'kB'}
                values += [f'ram_total_bytes={mem["MemTotal"]}', f'ram_available_bytes={mem["MemAvailable"]}',
                           f'ram_used_bytes={mem["MemTotal"] - mem["MemAvailable"]}',
                           f'swap_used_bytes={mem["SwapTotal"] - mem["SwapFree"]}']
            except (OSError, ValueError, IndexError, KeyError):
                values.append('system_cpu_ram=unavailable')
            torch.cuda.nvtx.mark('resources ' + ' '.join(values))
            current['time'] = now
            previous = current
            self.stop.wait(self.interval)

    def install_hooks(self, stage, torch):
        modules = [('model', stage.model)] + [(f'block_{i:02d}', b) for i, b in enumerate(stage.model.model.layers)]
        for name, module in modules:
            stack = []
            def before(module, inputs, name=name, stack=stack):
                label = name
                if name == 'model':
                    label = 'prefill' if self.forward_count == 0 else 'decode'
                    self.forward_count += 1
                torch.cuda.nvtx.range_push(label)
                stack.append(True)
            def after(module, inputs, output, stack=stack):
                if stack:
                    stack.pop()
                    torch.cuda.nvtx.range_pop()
            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after, always_call=True))

    @contextmanager
    def request(self, stage):
        import torch
        if self.used:
            raise ValueError('Use a fresh NsightBuffer per request.')
        self.used = True
        thread = None
        success = False
        try:
            self.install_hooks(stage, torch)
            if not self.monitor_active:
                thread = threading.Thread(target=self.monitor, args=(torch,), name='CPU_RAM_monitor', daemon=True)
                thread.start()
            start = time.perf_counter()
            try:
                yield self
                success = True
            finally:
                end = time.perf_counter()
                if success:
                    count = len(self.tokens)
                    ttft = (self.tokens[0] - start) * 1000 if count else None
                    tpot = (self.tokens[-1] - self.tokens[0]) * 1000 / (count - 1) if count > 1 else None
                    torch.cuda.nvtx.mark(f'application input_tokens={self.input_count} output_tokens={count} '
                                         f'latency_ms={(end-start)*1000:.3f} ttft_ms={ttft} tpot_ms={tpot} '
                                         f'output_tokens_per_second={count/(end-start):.3f}')
                else:
                    torch.cuda.nvtx.mark('request_failed: application_metrics_invalid')
        finally:
            if thread is not None and thread.ident is not None:
                self.stop.set()
                thread.join(timeout=2)
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
