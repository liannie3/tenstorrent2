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
