"""Regression checks for telemetry units, device identity and timing attribution."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from telemetry import TelemetryMonitor, parse_prometheus, parse_snapshot, snapshot_metrics, discover_sensors


class TelemetryTests(unittest.TestCase):
    def test_smi_units_and_multiple_boards(self):
        data = {'device_info': [
            {'board_info': {'bus_id': '01:00.0'}, 'telemetry': {
                'current': ' 12.5', 'asic_temperature': ' 61.2', 'board_power': 'N/A'}},
            {'board_info': {'bus_id': '02:00.0'}, 'telemetry': {'asic_temperature': 70}}]}
        values = snapshot_metrics(parse_snapshot('Progress...\n' + json.dumps(data)))
        self.assertIn(('current', 12.5, 'A'), [row[:3] for row in values])
        temps = [row for row in values if row[0] == 'asic_temperature']
        self.assertEqual([row[1] for row in temps], [61.2, 70])
        self.assertNotEqual(temps[0][3], temps[1][3])
        self.assertIn(('board_power', '', 'W'), [row[:3] for row in values])
        self.assertFalse(any('utilization' in row[0] for row in values))

    def test_prometheus_labels_and_nonfinite(self):
        values = parse_prometheus('''# HELP x example
 tt_asic_temperature_celsius 1
tt_asic_temperature_celsius{chip="0",tray="0"} 61.2
tt_asic_temperature_celsius{chip="1",tray="0"} NaN
tt_dispatch_utilization_percent{chip="0",cq="1"} 35
tt_device_dram_used_megabytes{chip="0"} 128
''')
        self.assertEqual(len(values), 3)
        self.assertEqual(values[1][2], '%')
        self.assertEqual(values[2][2], 'MiB')
        self.assertIn('chip="0"', values[0][3])

    def test_hwmon_units_and_device_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            device = root/'pci'/'0000:01:00.0'
            device.mkdir(parents=True)
            (device/'vendor').write_text('0x1e52')
            hwmon = root/'hwmon0'
            hwmon.mkdir()
            (hwmon/'device').symlink_to(device)
            (hwmon/'name').write_text('tenstorrent')
            (hwmon/'temp1_input').write_text('62000')
            (hwmon/'temp1_label').write_text('ASIC')
            sensors = discover_sensors(root)
            self.assertEqual(len(sensors), 1)
            self.assertEqual(float(sensors[0][0].read_text())/sensors[0][2], 62)
            self.assertIn('0000:01:00.0', sensors[0][3])

    def test_crossing_samples_excluded_from_summary(self):
        monitor = TelemetryMonitor()
        monitor.windows = [(1, 100, 200), (2, 201, 300)]
        for start, end, value in [(110,120,60), (190,210,100)]:
            monitor.emit('sensor', 'temperature', value, 'C', start=start)
            monitor.rows[-1]['read_end_ns'] = end
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'telemetry.csv'
            monitor.finish(path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['run'], '1')
            self.assertEqual(rows[1]['run'], '')
            self.assertEqual(json.loads(rows[1]['overlapping_runs']), [1,2])
            self.assertEqual(monitor.summary(1)[0]['mean'], 60)
            self.assertEqual(monitor.summary(2), [])

    def test_missing_tool_is_recorded(self):
        monitor = TelemetryMonitor(tt_smi=True)
        with patch('telemetry.subprocess.run', side_effect=FileNotFoundError('missing tt-smi')):
            monitor.device_sample()
        self.assertIn('missing tt-smi', monitor.rows[-1]['status'])

    def test_real_host_collection_and_cleanup(self):
        monitor = TelemetryMonitor(interval=0.02)
        with TemporaryDirectory() as tmp:
            monitor.start()
            monitor.begin_request(1)
            # Wait for polling activity only in this host-only regression test.
            import time
            time.sleep(0.08)
            cpu = monitor.end_request()
            monitor.finish(Path(tmp)/'host.csv')
        self.assertGreaterEqual(cpu, 0)
        self.assertTrue(any(row['metric']=='process_rss_bytes' for row in monitor.rows))
        self.assertTrue(all(not thread.is_alive() for thread in monitor.threads))

    def test_benchmark_integration(self):
        from contracts import StageAnswer
        from benchmark_latency import main
        class FakeStage:
            def __init__(self, *args, **kwargs): pass
            def answer(self, question, *, max_new_tokens, benchmark):
                if benchmark:
                    with benchmark.request(self):
                        benchmark.input_ready(4)
                        benchmark.token_ready()
                return StageAnswer('fixture', (42,), 'eos')
        import sys
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'latency.csv'
            args = ['benchmark_cpu.py', '--model','fixture','--text-only', '--runs','2',
                    '--warmup','0', '--output',str(output)]
            with patch.dict(sys.modules, tinyllama_stage=SimpleNamespace(TinyLlamaStage=FakeStage),
                            torch_xla=SimpleNamespace(sync=lambda wait: None)), patch.object(sys,'argv',args):
                main('cpu')
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(output.with_suffix('.telemetry.csv').exists())
            self.assertTrue(all(row['telemetry_enabled']=='True' and row['tpot_ms']=='' for row in rows))


if __name__ == '__main__':
    unittest.main()
