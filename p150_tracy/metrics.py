"""Request measurements use one monotonic clock. No accelerator dependencies."""
import csv
from contextlib import contextmanager, nullcontext
from pathlib import Path
import threading
import time


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class Observer:
    def __init__(self, mark=None):
        self.mark = mark
        self.spans = []
        self.tokens = []
        self.input_count = 0

    def request(self, model):
        return nullcontext()

    def input_ready(self, count):
        self.input_count = count

    def token_ready(self):
        self.tokens.append(time.perf_counter())

    @contextmanager
    def span(self, name):
        if self.mark:
            self.mark(header=f"begin_{name}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.spans.append(dict(stage=name, wall_ms=elapsed * 1000))
            if self.mark:
                self.mark(header=f"end_{name}")

    def timing(self, start, end):
        n = len(self.tokens)
        delta = self.tokens[-1] - self.tokens[0]
        return dict(input_tokens=self.input_count, output_tokens=n, latency_ms=(end-start)*1000,
                    ttft_ms=(self.tokens[0]-start)*1000,
                    tpot_ms=delta*1000/(n-1) if n > 1 else None,
                    decode_tokens_s=(n-1)/delta if n > 1 and delta > 0 else None,
                    request_tokens_s=n/(end-start))


def sensors(bdf):
    base = Path("/sys/bus/pci/devices")
    matches = [p for p in base.iterdir() if (p/"vendor").read_text().strip() == "0x1e52"]
    if bdf:
        matches = [p for p in matches if p.name == bdf]
    if len(matches) != 1:
        raise ValueError("Specify the P150 PCI bus address from tt-smi with --bdf DOMAIN:BUS:SLOT.FUNC")
    board = matches[0]
    paths = {}
    for key, pattern, scale in [
        ("power_w", "power1_input", 1e6), ("temp_c", "temp1_input", 1e3),
        ("aiclk_mhz", "tt_aiclk", 1)]:
        found = list(board.glob("**/" + pattern))
        if found:
            paths[key] = (found[0], scale)
    if not paths:
        raise RuntimeError(f"No telemetry attributes under {board}. Inspect tt-smi and TT-KMD sysfs.")
    return paths


class Sampler:
    def __init__(self, paths, interval):
        self.paths = paths
        self.interval = interval
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        row = dict(time_s=time.perf_counter())
        for key, (path, scale) in self.paths.items():
            try:
                row[key] = float(path.read_text().strip()) / scale
            except (OSError, ValueError):
                row[key] = None
        self.rows.append(row)

    def loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()


def energy(rows, start, end, interval):
    """Clip trapezoids to request bounds. Reject absent readings and long gaps."""
    joules = covered = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = b["time_s"] - a["time_s"]
        lo, hi = max(start, a["time_s"]), min(end, b["time_s"])
        if hi <= lo or dt <= 0 or dt > 3 * interval:
            continue
        p, q = a.get("power_w"), b.get("power_w")
        if p is None or q is None:
            continue
        slope = (q-p)/dt
        joules += (hi-lo)*(p + slope*((lo+hi)/2-a["time_s"]))
        covered += hi-lo
    inside = sum(start <= row["time_s"] <= end and row.get("power_w") is not None for row in rows)
    fraction = covered/(end-start)
    return (joules if inside >= 2 and fraction >= 0.999999 else None), fraction
