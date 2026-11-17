"""CPU-only parser and launcher tests; no Jetson hardware is required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tegrastats_monitor import TegrastatsMonitor, parse_tegrastats


class FakeNVTX:
    def __init__(self):
        self.events = []

    def mark(self, label):
        self.events.append(label)


class FakeCuda:
    def __init__(self):
        self.nvtx = FakeNVTX()


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class Parsing(unittest.TestCase):
    def test_orin_util_clocks_rails(self):
        line = ('RAM 1100/31950MB (lfb 200x4MB) SWAP 0/4096MB '
                'CPU [17%@729,off,33%@729] EMC_FREQ 24%@3199 '
                'GR3D_FREQ 88%@[1098,1098] GPU@48.5C CPU@42C SOC0@47.0C MCPU@46.0C '
                'VDD_IN 12000mW/11000mW VDD_CPU_GPU_CV 5000/4600')
        m = parse_tegrastats(line)
        self.assertEqual(m['ram_used_mb'], 1100)
        self.assertEqual(m['gpu_active_percent'], 88)
        self.assertEqual(m['gpu_clock_mhz'], 1098)
        self.assertEqual(m['emc_util_percent'], 24)
        self.assertEqual(m['emc_clock_mhz'], 3199)
        self.assertEqual(m['gpu_temp_c'], 48.5)
        self.assertEqual(m['soc0_temp_c'], 47.0)
        self.assertEqual(m['mcpu_temp_c'], 46.0)
        self.assertEqual(m['vdd_in_power_w'], 12)
        self.assertEqual(m['vdd_cpu_gpu_cv_power_w'], 5)
        self.assertEqual(m['cpu_online_core_mean_percent'], 25)
        self.assertNotIn('measured_memory_bandwidth_gbs', m)

    def test_missing_fields_not_zero(self):
        m = parse_tegrastats('RAM 100/200MB GR3D_FREQ @[1098,1098]')
        self.assertNotIn('gpu_active_percent', m)
        self.assertEqual(m['gpu_clock_mhz'], 1098)
        self.assertNotIn('vdd_in_power_w', m)

    def test_standalone_sample_monitor_marks_and_shutdown(self):
        torch = FakeTorch()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fake_tegrastats'
            # A stub command simulates actual streaming and stays alive until terminated.
            path.write_text('#!/usr/bin/env python3\n'
                'import time\n'
                'print("RAM 500/3000MB GR3D_FREQ 72%@900 GPU@49.5C CPU@43.0C SOC0@45.0C VDD_IN 11000/9000", flush=True)\n'
                'time.sleep(20)\n')
            path.chmod(0o755)
            with patch('tegrastats_monitor.shutil.which', return_value=str(path)):
                monitor = TegrastatsMonitor(100)
                monitor.start(torch)
                # Consume until the fake has supplied its first sample.
                import time
                for _ in range(500):
                    with monitor._lock:
                        if monitor.sample_count:
                            break
                    time.sleep(.01)
                monitor.stop(torch)
            self.assertEqual(monitor.sample_count, 1)
            self.assertIsNotNone(monitor.process.poll())
            labels = torch.cuda.nvtx.events
            self.assertTrue(any(x.startswith('tegrastats gpu_active_percent=72.000')
                                or 'gpu_active_percent=72.000' in x for x in labels))
            self.assertTrue(any('status=sampled' in x for x in labels))
            self.assertTrue(any('gpu_temp_c=49.500' in x for x in labels))
            self.assertTrue(any('gpu_temp_mean_c=49.500' in x and 'gpu_temp_max_c=49.500' in x for x in labels))
            self.assertTrue(any('soc0_temp_mean_c=45.000' in x for x in labels))


if __name__ == '__main__':
    unittest.main()
