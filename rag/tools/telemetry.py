"""Background accelerator telemetry sampling for benchmark trials."""

from __future__ import annotations

import json
import math
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any


SAMPLE_FIELDS = [
    "backend", "precision", "batch", "trial", "provider", "device",
    "timestamp_utc", "elapsed_s", "temperature_c", "power_w",
    "core_clock_mhz", "memory_clock_mhz", "utilization_pct",
    "thermal_limit_c", "throttled", "error",
]

SUMMARY_FIELDS = [
    "Platform", "Type", "Batch", "Provider", "Trials", "Samples",
    "Sampling errors", "Initial temp C", "Mean temp C", "Max temp C",
    "Final temp C", "Thermal rise C", "Thermal limit C",
    "Throttled samples", "Mean power W", "Max power W",
    "Mean core clock MHz", "Min core clock MHz", "Mean utilization %",
]


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
        if not match:
            return None
        result = float(match.group())
    return result if math.isfinite(result) else None


def nested(mapping: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def parse_nvidia_csv(text: str) -> dict[str, float | str | None]:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("nvidia-smi returned no telemetry row")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 7:
        raise ValueError("nvidia-smi returned an unexpected telemetry schema")
    (temperature, power, core_clock, memory_clock, utilization,
     software_thermal, hardware_thermal) = values
    inactive = {"0", "0x0000000000000000", "not active"}
    throttled = any(
        reason.strip() not in inactive
        for reason in (software_thermal, hardware_thermal)
    )
    return {
        "temperature_c": number(temperature),
        "power_w": number(power),
        "core_clock_mhz": number(core_clock),
        "memory_clock_mhz": number(memory_clock),
        "utilization_pct": number(utilization),
        "thermal_limit_c": None,
        "throttled": "1" if throttled else "0",
    }


def parse_tenstorrent_json(text: str, device: int) -> dict[str, float | str | None]:
    document = json.loads(text)
    devices = document.get("device_info")
    if devices is None:
        devices = document.get("devices")
    if isinstance(devices, dict):
        devices = list(devices.values())
    if not isinstance(devices, list) or device >= len(devices):
        # Some older snapshots contain a single device object at the top level.
        devices = [document]
    if device >= len(devices) or not isinstance(devices[device], dict):
        raise ValueError(f"tt-smi snapshot has no device {device}")
    item = devices[device]
    telemetry = nested(item, "telemetry")
    if not isinstance(telemetry, dict):
        telemetry = item
    limits = nested(item, "limits")
    if not isinstance(limits, dict):
        limits = {}

    temperature = number(nested(
        telemetry, "asic_temperature", "temperature", "core_temperature"
    ))
    limit = number(nested(
        limits, "asic_temperature", "temperature", "thermal_limit",
        "temperature_limit",
    ))
    power = number(nested(telemetry, "board_power", "power", "core_power"))
    clock = number(nested(telemetry, "aiclk", "ai_clock", "core_clock"))
    throttled = None
    if temperature is not None and limit is not None:
        throttled = "1" if temperature >= limit else "0"
    return {
        "temperature_c": temperature,
        "power_w": power,
        "core_clock_mhz": clock,
        "memory_clock_mhz": None,
        "utilization_pct": number(nested(telemetry, "utilization")),
        "thermal_limit_c": limit,
        "throttled": throttled,
    }


def parse_tegrastats(text: str) -> dict[str, float | str | None]:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("tegrastats returned no telemetry row")
    row = rows[-1]
    temperature_match = re.search(r"\bgpu@([0-9.]+)C", row, re.IGNORECASE)
    frequency_match = re.search(
        r"\bGR3D_FREQ\s+([0-9.]+)%?(?:@([0-9.]+))?", row, re.IGNORECASE
    )
    power_match = re.search(
        r"\bVDD_GPU(?:_SOC)?\s+([0-9.]+)mW", row, re.IGNORECASE
    )
    return {
        "temperature_c": number(temperature_match.group(1)) if temperature_match else None,
        "power_w": (
            number(power_match.group(1)) / 1000.0 if power_match else None
        ),
        "core_clock_mhz": (
            number(frequency_match.group(2))
            if frequency_match and frequency_match.group(2) else None
        ),
        "memory_clock_mhz": None,
        "utilization_pct": (
            number(frequency_match.group(1)) if frequency_match else None
        ),
        "thermal_limit_c": None,
        "throttled": None,
    }


def sample_once(provider: str, device: int) -> dict[str, float | str | None]:
    if provider == "nvidia":
        fields = (
            "temperature.gpu,power.draw,clocks.sm,clocks.mem,"
            "utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,"
            "clocks_event_reasons.hw_thermal_slowdown"
        )
        completed = subprocess.run(
            [
                "nvidia-smi", f"--id={device}", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return parse_nvidia_csv(completed.stdout)
    if provider == "jetson":
        process = subprocess.Popen(
            ["tegrastats", "--interval", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, _ = process.communicate(timeout=0.35)
        except subprocess.TimeoutExpired:
            process.terminate()
            stdout, _ = process.communicate(timeout=2)
        return parse_tegrastats(stdout)
    if provider == "tenstorrent":
        completed = subprocess.run(
            ["tt-smi", "-s"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return parse_tenstorrent_json(completed.stdout, device)
    raise ValueError(f"unsupported telemetry provider: {provider}")


class TelemetrySampler:
    def __init__(self, provider: str, device: int = 0, interval_ms: int = 500):
        if provider not in {"nvidia", "jetson", "tenstorrent"}:
            raise ValueError(f"unsupported telemetry provider: {provider}")
        if device < 0 or interval_ms < 100:
            raise ValueError("device must be non-negative and interval at least 100 ms")
        self.provider = provider
        self.device = device
        self.interval_s = interval_ms / 1000.0
        self.samples: list[dict[str, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler already started")
        self._start = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, str]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                raise RuntimeError("telemetry sampler did not stop")
        return self.samples

    def _run(self) -> None:
        while not self._stop.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            elapsed = time.monotonic() - self._start
            row = {field: "" for field in SAMPLE_FIELDS}
            row.update({
                "provider": self.provider,
                "device": str(self.device),
                "timestamp_utc": timestamp,
                "elapsed_s": f"{elapsed:.6f}",
            })
            try:
                metrics = sample_once(self.provider, self.device)
                for key, value in metrics.items():
                    row[key] = "" if value is None else str(value)
            except Exception as error:  # Sampling failure must not kill the benchmark.
                row["error"] = f"{type(error).__name__}: {error}"
            self.samples.append(row)
            self._stop.wait(self.interval_s)


def summarize(samples: list[dict[str, str]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for sample in samples:
        key = (
            sample["backend"], sample["precision"], sample["batch"],
            sample["provider"],
        )
        groups.setdefault(key, []).append(sample)

    summaries = []
    for (backend, precision, batch, provider), rows in groups.items():
        successful = [row for row in rows if not row["error"]]

        def values(field: str) -> list[float]:
            return [float(row[field]) for row in successful if row[field] != ""]

        temperatures = values("temperature_c")
        powers = values("power_w")
        clocks = values("core_clock_mhz")
        utilization = values("utilization_pct")
        limits = values("thermal_limit_c")
        row: dict[str, str | int] = {field: "" for field in SUMMARY_FIELDS}
        row.update({
            "Platform": backend,
            "Type": precision.upper(),
            "Batch": batch,
            "Provider": provider,
            "Trials": len({item["trial"] for item in rows}),
            "Samples": len(successful),
            "Sampling errors": len(rows) - len(successful),
            "Throttled samples": sum(item["throttled"] == "1" for item in successful),
        })
        if temperatures:
            row.update({
                "Initial temp C": f"{temperatures[0]:.3f}",
                "Mean temp C": f"{sum(temperatures) / len(temperatures):.3f}",
                "Max temp C": f"{max(temperatures):.3f}",
                "Final temp C": f"{temperatures[-1]:.3f}",
                "Thermal rise C": f"{temperatures[-1] - temperatures[0]:.3f}",
            })
        if limits:
            row["Thermal limit C"] = f"{min(limits):.3f}"
        if powers:
            row["Mean power W"] = f"{sum(powers) / len(powers):.3f}"
            row["Max power W"] = f"{max(powers):.3f}"
        if clocks:
            row["Mean core clock MHz"] = f"{sum(clocks) / len(clocks):.3f}"
            row["Min core clock MHz"] = f"{min(clocks):.3f}"
        if utilization:
            row["Mean utilization %"] = f"{sum(utilization) / len(utilization):.3f}"
        summaries.append(row)
    return summaries
