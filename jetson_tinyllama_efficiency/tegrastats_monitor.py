"""Optional Jetson tegrastats reader. Observations become Nsight NVTX marks.

No benchmark report is written here; nsys retains the NVTX events in .nsys-rep.
This monitor observes *whole-device* telemetry, not TinyLlama-only power/usage.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from statistics import fmean


# ==================== TELEMETRY PARSING ====================
# Example formats in NVIDIA's tegrastats guide vary by JetPack and module.
_NUMBER = r"\d+(?:\.\d+)?"


def parse_tegrastats(line: str) -> dict[str, float]:
    """Parse supported fields only; unavailable fields are absent, never zero."""
    metrics: dict[str, float] = {}
    for label, prefix in (("RAM", "ram"), ("SWAP", "swap")):
        match = re.search(rf"\b{label}\s+({_NUMBER})/({_NUMBER})MB\b", line)
        if match:
            metrics[f"{prefix}_used_mb"] = float(match.group(1))
            metrics[f"{prefix}_total_mb"] = float(match.group(2))

    # GR3D_FREQ is GPU active-time percentage, NOT process GPU utilization.
    # Frequency can appear as @1098 or @[1098,1098] on Orin.
    for label, key in (("GR3D_FREQ", "gpu_active_percent"),
                       ("EMC_FREQ", "emc_util_percent"),
                       ("EMC", "emc_util_percent")):
        match = re.search(rf"\b{label}\s+({_NUMBER})%", line)
        if match and key not in metrics:
            metrics[key] = float(match.group(1))

    for label, key in (("GR3D_FREQ", "gpu_clock_mhz"),
                       ("EMC_FREQ", "emc_clock_mhz"),
                       ("EMC", "emc_clock_mhz")):
        # Match both single frequencies and bracketed per-GPC values.
        match = re.search(rf"\b{label}\s+(?:{_NUMBER}%\s*)?@\[?({_NUMBER})", line)
        if match and key not in metrics:
            metrics[key] = float(match.group(1))

    # Jetson thermal-zone names vary: GPU, CPU, SOC0, SOC2, MCPU, etc.
    # Preserve the sensor name; only the `name@numberC` pattern is a temperature.
    for match in re.finditer(rf"\b([A-Za-z][A-Za-z0-9_-]*)@(-?{_NUMBER})C\b", line):
        sensor = match.group(1).lower()
        metrics[f"{sensor}_temp_c"] = float(match.group(2))

    # Do not indiscriminately sum named rails: some overlap in coverage.
    # For a rail "VDD_IN 12000mW/11000mW", record CURRENT power as watts.
    for match in re.finditer(
        rf"\b((?:VDD|VIN)[A-Z0-9_]*)\s+({_NUMBER})(?:mW)?/({_NUMBER})(?:mW)?\b", line
    ):
        name = match.group(1).lower()
        metrics[f"{name}_power_w"] = float(match.group(2)) / 1000.0
        metrics[f"{name}_reported_average_w"] = float(match.group(3)) / 1000.0

    # ==================== AGX ORIN MONITORED RAIL GROUP ====================
    # NVIDIA R36.2: VIN_SYS_5V0 includes VDDQ_VDD2_1V8AO. Do not add that
    # nested rail or any summation channel. This is not wall-plug power.
    group = ("vdd_gpu_soc_power_w", "vdd_cpu_cv_power_w", "vin_sys_5v0_power_w")
    if all(key in metrics for key in group):
        metrics["agx_monitored_rails_power_w"] = sum(metrics[key] for key in group)

    # ==================== CPU CLOCKS / UTILIZATION ====================
    cpu_match = re.search(r"\bCPU\s*\[([^]]+)\]", line)
    if cpu_match:
        loads = [float(m.group(1)) for part in cpu_match.group(1).split(",")
                 if (m := re.search(rf"({_NUMBER})%", part))]
        if loads:
            metrics["cpu_online_core_mean_percent"] = fmean(loads)
            metrics["cpu_online_core_count"] = float(len(loads))
        clocks = []
        for index, part in enumerate(cpu_match.group(1).split(",")):
            match = re.search(rf"({_NUMBER})%\s*@({_NUMBER})", part)
            if match:
                load, clock = map(float, match.groups())
                metrics[f"cpu_core_{index}_active_percent"] = load
                metrics[f"cpu_core_{index}_clock_mhz"] = clock
                clocks.append(clock)
        if clocks:
            metrics["cpu_online_core_mean_clock_mhz"] = fmean(clocks)

    # Preserve individual GPU GPC frequencies instead of treating the first as all.
    gpc = re.search(r"\bGR3D_FREQ\s+(?:[\d.]+%\s*)?@\[([^]]+)\]", line)
    if gpc:
        for index, value in enumerate(gpc.group(1).split(",")):
            if re.fullmatch(_NUMBER, value.strip()):
                metrics[f"gpu_gpc_{index}_clock_mhz"] = float(value)
    if "ram_used_mb" in metrics and metrics.get("ram_total_mb", 0) > 0:
        metrics["ram_used_percent"] = 100 * metrics["ram_used_mb"] / metrics["ram_total_mb"]

    return metrics


# ==================== ENERGY / EFFICIENCY CALCULATION ====================
def estimate_energy(points, start, end, interval_s):
    """Integrate current power samples using host receipt timestamps.

    Only use points inside the synchronized request. Trapezoidal integration
    between points; nearest sample held at each end for a bounded estimate.
    Reject sparse coverage or large gaps instead of inventing energy values.
    Returns diagnostics even when no request energy estimate is valid.
    """
    points = sorted((t, p) for t, p in points if start <= t <= end)
    duration = end - start
    result = {"power_samples": len(points)}
    if duration <= 0 or len(points) < 2:
        return dict(result, status="insufficient_samples")
    covered = points[-1][0] - points[0][0]
    gaps = [b[0] - a[0] for a, b in zip(points, points[1:])]
    head, tail = points[0][0] - start, end - points[-1][0]
    result.update(coverage_percent=100 * covered / duration,
                  max_gap_s=max(gaps), start_gap_s=head, end_gap_s=tail)
    # Conservative project policy, not a NVIDIA sensor accuracy specification.
    if covered / duration < 0.8 or max(gaps) > 2.5 * interval_s or max(head, tail) > 2 * interval_s:
        return dict(result, status="insufficient_coverage")
    observed = sum((a[1] + b[1]) * (b[0] - a[0]) / 2
                   for a, b in zip(points, points[1:]))
    energy = observed + points[0][1] * head + points[-1][1] * tail
    return dict(result, status="estimated", energy_j=energy,
                observed_span_energy_j=observed, time_weighted_mean_w=energy / duration)


class TegrastatsMonitor:
    """Read tegrastats in a daemon thread; emit compact NVTX instant events."""

    def __init__(self, interval_ms: int = 100):
        if interval_ms < 20:
            raise ValueError("tegrastats interval must be >= 20 ms")
        self.interval_ms = interval_ms
        self.process = None
        self.thread = None
        self.sample_count = 0
        # ==================== TIMESTAMPED HARDWARE SAMPLES ====================
        self.samples = []  # (host receipt time, parsed fields)
        self.request_window = None
        self.output_tokens = None
        self.request_success = False
        self.error = None
        self._lock = threading.Lock()

    def start(self, torch):
        exe = shutil.which("tegrastats")
        if not exe:
            self.error = "tegrastats_not_found"
            torch.cuda.nvtx.mark("tegrastats_status unavailable_reason=not_found")
            return
        try:
            self.process = subprocess.Popen(
                [exe, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError as exc:
            self.error = f"start_failed_{type(exc).__name__}"
            torch.cuda.nvtx.mark(f"tegrastats_status {self.error}")
            return
        self.thread = threading.Thread(
            target=self._read_lines, args=(torch,), name="tegrastats_reader", daemon=True
        )
        self.thread.start()
        torch.cuda.nvtx.mark(f"tegrastats_status started interval_ms={self.interval_ms}")

    def set_request_window(self, start, end, output_tokens, *, success):
        # Use the same timestamps as pipeline_latency_ms, not subprocess lifetime.
        self.request_window = (start, end)
        self.output_tokens = output_tokens
        self.request_success = success

    def _read_lines(self, torch):
        try:
            for line in self.process.stdout:
                received = time.perf_counter()
                fields = parse_tegrastats(line)
                if not fields:
                    continue
                with self._lock:
                    self.sample_count += 1
                    self.samples.append((received, fields))
                # This NVTX event timestamps receipt/parse, not the sensor's hardware time.
                # Event appears on the tegrastats_reader host thread in Nsight.
                record = " ".join(f"{key}={value:.3f}" for key, value in fields.items())
                torch.cuda.nvtx.mark("tegrastats " + record)
        except Exception as exc:
            self.error = f"reader_failed_{type(exc).__name__}"

    def stop(self, torch):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()  # Only stop our subprocess, not system-wide instances.
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            if self.thread is not None:
                self.thread.join(timeout=2)
            if self.process.stdout is not None:
                self.process.stdout.close()
        # ==================== REQUEST-ALIGNED SUMMARY ====================
        with self._lock:
            records = list(self.samples)
            if self.request_window is not None:
                start, end = self.request_window
                records = [(t, fields) for t, fields in records if start <= t <= end]
            samples = len(records)
            series = {}
            for t, fields in records:
                for key, value in fields.items():
                    series.setdefault(key, []).append((t, value))
            util = fmean(v for _, v in series["gpu_active_percent"]) if "gpu_active_percent" in series else None
            rail_summary = {key: fmean(v for _, v in values)
                            for key, values in series.items() if key.endswith("_power_w")}
            temperature_summary = {
                key: (fmean(v for _, v in values), min(v for _, v in values),
                      max(v for _, v in values), len(values))
                for key, values in series.items() if key.endswith("_temp_c")
            }
        # ==================== TEMPERATURE / POWER / CLOCK STATISTICS ====================
        parts = [f"sample_count={samples}"]
        if util is not None:
            parts.append(f"gpu_active_mean_percent={util:.3f}")
        for key, value in sorted(rail_summary.items()):
            values = [v for _, v in series[key]]
            stem = key.removesuffix("_power_w")
            parts.extend((f"mean_{key}={value:.3f}",
                          f"{stem}_power_min_w={min(values):.3f}",
                          f"{stem}_power_peak_w={max(values):.3f}"))
        for key, (mean_c, min_c, max_c, count) in sorted(temperature_summary.items()):
            stem = key.removesuffix("_c")
            parts.extend((f"{stem}_mean_c={mean_c:.3f}",
                          f"{stem}_min_c={min_c:.3f}",
                          f"{stem}_max_c={max_c:.3f}",
                          f"{stem}_samples={count}"))
            if count >= 2:
                parts.append(f"{stem}_change_c={series[key][-1][1] - series[key][0][1]:.3f}")
        # Generic summaries for utilization, clock and memory values (sample statistics).
        for key, observations in sorted(series.items()):
            if key.endswith(("_clock_mhz", "_percent", "_used_mb")):
                values = [v for _, v in observations]
                parts.extend((f"mean_{key}={fmean(values):.3f}",
                              f"min_{key}={min(values):.3f}",
                              f"max_{key}={max(values):.3f}"))
        if self.error:
            parts.append(f"status={self.error}")
        elif samples == 0:
            parts.append("status=no_samples")
        else:
            parts.append("status=sampled")
        torch.cuda.nvtx.mark("tegrastats_summary " + " ".join(parts))

        # ==================== POWER EFFICIENCY NVTX EVENTS ====================
        if self.request_window is None:
            return
        start, end = self.request_window
        if not self.request_success or self.error or not rail_summary:
            reason = "request_failed" if not self.request_success else (self.error or "no_power_samples")
            torch.cuda.nvtx.mark(f"energy_efficiency status={reason}")
            return
        for key in sorted(rail_summary):
            rail = key.removesuffix("_power_w")
            result = estimate_energy(series[key], start, end, self.interval_ms / 1000)
            fields = [f"rail={rail}", f"request_s={end-start:.6f}",
                      f"output_tokens={self.output_tokens}"]
            for k, v in result.items():
                fields.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
            if result["status"] == "estimated":
                energy = result["energy_j"]
                fields.append(f"energy_delay_j_s={energy * (end-start):.6f}")
                if self.output_tokens is not None and self.output_tokens > 0 and energy > 0:
                    fields.extend((f"joules_per_token={energy/self.output_tokens:.6f}",
                                   f"tokens_per_joule={self.output_tokens/energy:.6f}"))
            torch.cuda.nvtx.mark("energy_efficiency " + " ".join(fields))
