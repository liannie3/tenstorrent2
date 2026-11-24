"""Background accelerator telemetry sampling for benchmark trials."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.request import urlopen


SAMPLE_FIELDS = [
    "backend", "precision", "batch", "trial", "provider", "device",
    "timestamp_utc", "elapsed_s", "temperature_c", "power_w",
    "core_clock_mhz", "memory_clock_mhz", "utilization_pct",
    "thermal_limit_c", "throttled", "power_rail", "power_rails_w", "phase", "error",
]

POWER_RAIL_FIELDS = [
    "backend", "precision", "batch", "trial", "provider", "device",
    "timestamp_utc", "elapsed_s", "phase", "rail", "power_w",
]

SUMMARY_FIELDS = [
    "Platform", "Type", "Batch", "Provider", "Power rail", "Trials", "Samples",
    "Sampling errors", "Initial temp C", "Mean temp C", "Max temp C",
    "Final temp C", "Thermal rise C", "Thermal limit C",
    "Throttled samples", "Mean power W", "Max power W",
    "Mean core clock MHz", "Min core clock MHz", "Mean utilization %",
    "Measured duration s", "Measured energy J", "Max sample gap s",
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


def parse_prometheus_labels(encoded: str) -> dict[str, str]:
    return {
        match.group(1): bytes(match.group(2), "utf-8").decode("unicode_escape")
        for match in re.finditer(r'(\w+)="((?:\\.|[^"\\])*)"', encoded)
    }


def parse_tt_telemetry(
    text: str, device: int, power_rail: str = "unspecified"
) -> dict[str, float | str | None]:
    """Parse one device from a tt-telemetry Prometheus metrics snapshot."""
    series: dict[str, list[tuple[dict[str, str], float]]] = {}
    pattern = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match is None:
            continue
        value = float(match.group(3))
        if math.isfinite(value):
            series.setdefault(match.group(1), []).append((
                parse_prometheus_labels(match.group(2) or ""), value
            ))

    identity_rows = series.get("tt_asic_temperature_celsius", [])
    identities = sorted({
        (labels.get("tray", ""), labels.get("chip", ""))
        for labels, _ in identity_rows
    })
    if device >= len(identities):
        raise ValueError(f"tt-telemetry snapshot has no device {device}")
    tray, chip = identities[device]

    def metric(*names: str) -> float | None:
        for name in names:
            for labels, value in series.get(name, []):
                if labels.get("tray", "") == tray and labels.get("chip", "") == chip:
                    return value
        return None

    power_rails = {
        name: value
        for name, rows in series.items()
        if name.lower().endswith(("_w", "_watts"))
        for labels, value in rows
        if labels.get("tray", "") == tray and labels.get("chip", "") == chip
    }
    selected_rails, selected_power = select_power_rails(
        power_rails, power_rail, ("tt_tdp_w",)
    )

    temperature = metric("tt_asic_temperature_celsius")
    limit = metric("tt_thermal_throttle_limit_celsius")
    throttled = None
    if temperature is not None and limit is not None:
        throttled = "1" if temperature >= limit else "0"
    return {
        "temperature_c": temperature,
        "power_w": selected_power,
        "power_rail": "+".join(selected_rails),
        "power_rails_w": json.dumps(
            dict(sorted(power_rails.items())), separators=(",", ":")
        ),
        "core_clock_mhz": metric("tt_ai_clock_mhz"),
        "memory_clock_mhz": metric("tt_dram_speed_mbps"),
        "utilization_pct": metric(
            "tt_dispatch_core_efficiency_percent",
            "tt_dispatch_utilization_percent",
        ),
        "thermal_limit_c": limit,
        "throttled": throttled,
    }


def select_power_rails(
    rails: dict[str, float],
    requested: str,
    defaults: tuple[str, ...],
) -> tuple[list[str], float | None]:
    """Select one or more named rails and return their summed power."""

    if requested == "unspecified":
        selected = [name for name in defaults if name in rails][:1]
    else:
        selected = [name.strip() for name in requested.split(",") if name.strip()]
    missing = [name for name in selected if name not in rails]
    if missing:
        available = ", ".join(sorted(rails)) or "none"
        raise ValueError(
            f"power rail(s) not reported: {', '.join(missing)}; "
            f"available rails: {available}"
        )
    return selected, sum(rails[name] for name in selected) if selected else None


def parse_tegrastats(
    text: str, power_rail: str = "unspecified"
) -> dict[str, float | str | None]:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("tegrastats returned no telemetry row")
    row = rows[-1]
    temperature_match = re.search(r"\bgpu@([0-9.]+)C", row, re.IGNORECASE)
    frequency_match = re.search(
        r"\bGR3D_FREQ\s+([0-9.]+)%?(?:@([0-9.]+))?", row, re.IGNORECASE
    )
    rails = {
        match.group(1).upper(): number(match.group(2))
        for match in re.finditer(
            r"\b([A-Z][A-Z0-9_]*)\s+([0-9.]+)mW(?:/[0-9.]+mW)?",
            row,
            re.IGNORECASE,
        )
    }
    requested_rails = (
        ",".join(item.strip().upper() for item in power_rail.split(","))
        if power_rail != "unspecified" else power_rail
    )
    selected_rails, selected_power = select_power_rails(
        rails, requested_rails, ("VDD_GPU_SOC", "VDD_GPU")
    )
    return {
        "temperature_c": number(temperature_match.group(1)) if temperature_match else None,
        "power_w": selected_power / 1000.0 if selected_power is not None else None,
        "power_rail": "+".join(selected_rails),
        "power_rails_w": json.dumps(
            {
                name: value / 1000.0
                for name, value in sorted(rails.items())
                if value is not None
            },
            separators=(",", ":"),
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


def sample_once(
    provider: str, device: int, power_rail: str = "unspecified"
) -> dict[str, float | str | None]:
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
        return parse_tegrastats(stdout, power_rail)
    if provider == "tenstorrent":
        endpoint = os.environ.get(
            "TT_TELEMETRY_ENDPOINT", "http://127.0.0.1:8080/metrics"
        )
        with urlopen(endpoint, timeout=5) as response:
            snapshot = response.read().decode("utf-8")
        return parse_tt_telemetry(snapshot, device, power_rail)
    if provider == "tenstorrent-smi":
        completed = subprocess.run(
            ["tt-smi", "-s"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return parse_tenstorrent_json(completed.stdout, device)
    raise ValueError(f"unsupported telemetry provider: {provider}")


class TelemetrySampler:
    def __init__(
        self, provider: str, device: int = 0, interval_ms: int = 500,
        power_rail: str = "unspecified",
    ):
        if provider not in {
            "nvidia", "jetson", "tenstorrent", "tenstorrent-smi"
        }:
            raise ValueError(f"unsupported telemetry provider: {provider}")
        if device < 0 or interval_ms < 100:
            raise ValueError("device must be non-negative and interval at least 100 ms")
        self.provider = provider
        self.device = device
        self.interval_s = interval_ms / 1000.0
        self.power_rail = power_rail
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
                "power_rail": self.power_rail,
                "phase": "process",
            })
            try:
                metrics = sample_once(
                    self.provider, self.device, self.power_rail
                )
                for key, value in metrics.items():
                    row[key] = "" if value is None else str(value)
            except Exception as error:  # Sampling failure must not kill the benchmark.
                row["error"] = f"{type(error).__name__}: {error}"
            self.samples.append(row)
            self._stop.wait(self.interval_s)


def mark_measured_phase(
    samples: list[dict[str, str]], measured_duration_s: float
) -> None:
    """Mark samples belonging to the final measured phase of rag_bench.

    The measured loop is the final long-running operation in rag_bench. Samples
    are therefore selected backwards from the final successful sample. Runs too
    short to contain two measured samples remain unmarked rather than yielding a
    misleading energy value.
    """
    successful = [row for row in samples if not row["error"]]
    if len(successful) < 2 or measured_duration_s <= 0:
        return
    finish = float(successful[-1]["elapsed_s"])
    start = max(0.0, finish - measured_duration_s)
    measured = [
        row for row in successful if float(row["elapsed_s"]) >= start
    ]
    if len(measured) < 2:
        return
    for row in measured:
        row["phase"] = "measured"


def integrate_energy(samples: list[dict[str, str]]) -> tuple[float, float] | None:
    """Return measured duration and trapezoid-integrated joules."""
    points = sorted(
        (
            (float(row["elapsed_s"]), float(row["power_w"]))
            for row in samples
            if row["phase"] == "measured" and row["power_w"] != ""
            and not row["error"]
        ),
        key=lambda point: point[0],
    )
    if len(points) < 2:
        return None
    energy = sum(
        (right_time - left_time) * (left_power + right_power) / 2.0
        for (left_time, left_power), (right_time, right_power)
        in zip(points, points[1:])
    )
    return points[-1][0] - points[0][0], energy


def expand_power_rails(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Expand time-aligned Jetson rail JSON into one CSV row per rail."""
    rows = []
    for sample in samples:
        encoded = sample.get("power_rails_w", "")
        if not encoded or sample.get("error", ""):
            continue
        try:
            rails = json.loads(encoded)
        except (TypeError, ValueError):
            continue
        if not isinstance(rails, dict):
            continue
        for rail, power in sorted(rails.items()):
            row = {field: "" for field in POWER_RAIL_FIELDS}
            for field in POWER_RAIL_FIELDS:
                if field in sample:
                    row[field] = sample[field]
            row["rail"] = str(rail)
            row["power_w"] = str(power)
            rows.append(row)
    return rows


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
            "Power rail": next(
                (item["power_rail"] for item in rows if item["power_rail"]), ""
            ),
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
        trial_groups = {
            trial: [item for item in rows if item["trial"] == trial]
            for trial in {item["trial"] for item in rows}
        }
        measured_trials = [
            result for result in (
                integrate_energy(trial_rows)
                for trial_rows in trial_groups.values()
            ) if result is not None
        ]
        if measured_trials:
            duration = sum(item[0] for item in measured_trials)
            energy = sum(item[1] for item in measured_trials)
            row["Measured duration s"] = f"{duration:.6f}"
            row["Measured energy J"] = f"{energy:.6f}"
        gaps = []
        for trial_rows in trial_groups.values():
            elapsed = sorted(
                float(item["elapsed_s"])
                for item in trial_rows if not item["error"]
            )
            gaps.extend(
                right - left for left, right in zip(elapsed, elapsed[1:])
            )
        if gaps:
            max_gap = max(gaps)
            row["Max sample gap s"] = f"{max_gap:.6f}"
        summaries.append(row)
    return summaries
