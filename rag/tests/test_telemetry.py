"""Host-only parser and aggregation tests for accelerator telemetry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module(repo: Path):
    path = repo / "tools" / "telemetry.py"
    spec = importlib.util.spec_from_file_location("telemetry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    telemetry = load_module(Path(__file__).resolve().parents[1])
    nvidia = telemetry.parse_nvidia_csv(
        "67, 121.5, 1095, 6250, 88, Not Active, 0x0000000000000000\n"
    )
    assert nvidia == {
        "temperature_c": 67.0,
        "power_w": 121.5,
        "core_clock_mhz": 1095.0,
        "memory_clock_mhz": 6250.0,
        "utilization_pct": 88.0,
        "thermal_limit_c": None,
        "throttled": "0",
    }
    jetson = telemetry.parse_tegrastats(
        "RAM 100/1000MB GR3D_FREQ 72%@918 gpu@61.5C "
        "VDD_GPU_SOC 12345mW/12000mW\n"
    )
    assert jetson["temperature_c"] == 61.5
    assert jetson["power_w"] == 12.345
    assert jetson["core_clock_mhz"] == 918.0
    assert jetson["utilization_pct"] == 72.0
    multi_rail = (
        "gpu@62C VDD_GPU_SOC 12000mW/11000mW "
        "VDD_CPU_CV 4500mW/4000mW VIN_SYS_5V0 22000mW/21000mW\n"
    )
    assert telemetry.parse_tegrastats(
        multi_rail, "VDD_CPU_CV"
    )["power_w"] == 4.5
    assert telemetry.parse_tegrastats(
        multi_rail, "vin_sys_5v0"
    )["power_w"] == 22.0
    all_rails = telemetry.parse_tegrastats(multi_rail)["power_rails_w"]
    assert all_rails == (
        '{"VDD_CPU_CV":4.5,"VDD_GPU_SOC":12.0,"VIN_SYS_5V0":22.0}'
    )
    try:
        telemetry.parse_tegrastats(multi_rail, "VDD_MISSING")
    except ValueError as error:
        assert "available rails" in str(error)
    else:
        raise AssertionError("missing tegrastats rail was accepted")

    rail_sample = {field: "" for field in telemetry.SAMPLE_FIELDS}
    rail_sample.update({
        "backend": "cuda-optimized", "precision": "fp32", "batch": "32",
        "trial": "1", "provider": "jetson", "device": "0",
        "timestamp_utc": "2026-11-10T00:00:00+00:00", "elapsed_s": "0.5",
        "phase": "measured", "power_rails_w": all_rails,
    })
    expanded = telemetry.expand_power_rails([rail_sample])
    assert [row["rail"] for row in expanded] == [
        "VDD_CPU_CV", "VDD_GPU_SOC", "VIN_SYS_5V0"
    ]
    assert [row["power_w"] for row in expanded] == ["4.5", "12.0", "22.0"]

    tt = telemetry.parse_tenstorrent_json(
        '{"device_info":[{"telemetry":{"asic_temperature":"72.5",'
        '"board_power":"188.2", "aiclk":"1250"},'
        '"limits":{"temperature":"90"}}]}',
        0,
    )
    assert tt["temperature_c"] == 72.5
    assert tt["power_w"] == 188.2
    assert tt["core_clock_mhz"] == 1250.0
    assert tt["thermal_limit_c"] == 90.0
    assert tt["throttled"] == "0"

    samples = []
    for trial, temperature, power in (("1", "60", "100"), ("1", "70", "120")):
        row = {field: "" for field in telemetry.SAMPLE_FIELDS}
        row.update({
            "backend": "cuda-optimized", "precision": "fp32", "batch": "32",
            "trial": trial, "provider": "nvidia", "temperature_c": temperature,
            "power_w": power, "core_clock_mhz": "1000", "throttled": "0",
        })
        samples.append(row)
    summary = telemetry.summarize(samples)[0]
    assert summary["Samples"] == 2
    assert summary["Mean temp C"] == "65.000"
    assert summary["Max temp C"] == "70.000"
    assert summary["Thermal rise C"] == "10.000"
    assert summary["Mean power W"] == "110.000"

    energy_samples = []
    for elapsed, power in (("0", "100"), ("1", "120"), ("2", "100")):
        row = {field: "" for field in telemetry.SAMPLE_FIELDS}
        row.update({
            "backend": "cuda-optimized", "precision": "fp32", "batch": "32",
            "trial": "1", "provider": "nvidia", "elapsed_s": elapsed,
            "power_w": power, "phase": "process",
        })
        energy_samples.append(row)
    telemetry.mark_measured_phase(energy_samples, 2.0)
    duration, energy = telemetry.integrate_energy(energy_samples)
    assert duration == 2.0
    assert energy == 220.0
