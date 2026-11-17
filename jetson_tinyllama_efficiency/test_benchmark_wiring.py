"""CPU-only tests for the links added to the user's benchmark scripts.

No model weights, Jetson hardware, or CUDA-enabled PyTorch are required.
These tests verify argument forwarding and monitor start/stop, not GPU timings.
"""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import benchmark_nsight
import benchmark_stages
import run
import run_all
from stage_observer import StageObserver


class _FakeNVTX:
    def __init__(self):
        self.events = []

    def mark(self, label):
        self.events.append(label)

    def range(self, name):
        self.events.append('range ' + name)
        return nullcontext()


class _FakeMonitor:
    instances = []

    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.active = False
        self.start_count = 0
        self.stop_count = 0
        self.instances.append(self)

    def start(self, torch):
        self.start_count += 1
        self.active = True
        torch.cuda.nvtx.mark('tegrastats_status started interval_ms=' + str(self.interval_ms))

    def set_request_window(self, start, end, output_tokens, *, success):
        self.window = (start, end, output_tokens, success)

    def stop(self, torch):
        self.stop_count += 1
        self.active = False
        torch.cuda.nvtx.mark('tegrastats_summary sample_count=0 status=synthetic')


class Tests(unittest.TestCase):
    def test_cli_parsers_accept_temperature_flags(self):
        args = run.build_parser().parse_args([
            '--model', 'models/tinyllama', '--text-only', '--question', 'hello',
            '--benchmark', 'results/test.nsys-rep', '--tegrastats',
            '--tegrastats-interval-ms', '250',
        ])
        self.assertTrue(args.tegrastats)
        self.assertEqual(args.tegrastats_interval_ms, 250)
        for mode in ('latency', 'cpu', 'gpu'):
            opts = benchmark_stages.parser_for(mode).parse_args([
                '--model', 'models/tinyllama', '--text-only',
                '--output', 'results/test.nsys-rep', '--tegrastats',
                '--tegrastats-interval-ms', '250',
            ])
            self.assertEqual(opts.test, mode)
            self.assertTrue(opts.tegrastats)
            self.assertEqual(opts.tegrastats_interval_ms, 250)

    def test_stage_observer_starts_and_stops_temperature_monitor(self):
        from nsight_buffer import NsightBuffer
        nvtx = _FakeNVTX()
        cuda = types.SimpleNamespace(nvtx=nvtx, synchronize=lambda _device: None)
        fake_torch = types.SimpleNamespace(cuda=cuda)
        stage = types.SimpleNamespace(device='cuda:0')
        _FakeMonitor.instances.clear()
        observer = StageObserver('cpu', stage.device, fake_torch,
                                 interval_ms=30, tegrastats=True,
                                 tegrastats_interval_ms=175)
        self.assertIsInstance(observer, NsightBuffer)
        with (patch.dict(sys.modules, {'torch': fake_torch}),
              patch('nsight_buffer.TegrastatsMonitor', _FakeMonitor)):
            with observer.pipeline_scope(stage, 'request_1'):
                time.sleep(.07)
        self.assertEqual(len(_FakeMonitor.instances), 1)
        monitor = _FakeMonitor.instances[0]
        self.assertEqual(monitor.interval_ms, 175)
        self.assertEqual((monitor.start_count, monitor.stop_count), (1, 1))
        self.assertGreater(monitor.window[1], monitor.window[0])
        self.assertTrue(monitor.window[3])
        self.assertTrue(any(x.startswith('pipeline_latency_ms=') for x in nvtx.events))
        self.assertTrue(any('tegrastats_summary' in x for x in nvtx.events))

    def test_separate_launcher_forwards_tegrastats_into_worker(self):
        args = benchmark_stages.parser_for('gpu').parse_args([
            '--model', 'models/tinyllama', '--text-only',
            '--output', 'unused.nsys-rep', '--tegrastats',
            '--interval-ms', '80', '--tegrastats-interval-ms', '190'
        ])
        with tempfile.TemporaryDirectory() as temp:
            args.output = str(Path(temp) / 'capture.nsys-rep')
            got = []

            def fake_run(cmd, **_):
                got.extend(cmd)
                prefix = next(x[len('--output='):] for x in cmd if x.startswith('--output='))
                Path(prefix + '.nsys-rep').write_bytes(b'nonempty test report')
                return types.SimpleNamespace(returncode=0)

            with patch('benchmark_stages.shutil.which', return_value='/usr/bin/nsys'), \
                 patch('benchmark_stages.subprocess.run', side_effect=fake_run):
                self.assertEqual(benchmark_stages.launch(args), 0)
            self.assertTrue(Path(args.output).exists())
            self.assertIn('--tegrastats', got)
            self.assertIn('--tegrastats-interval-ms', got)
            self.assertIn('190', got)
            self.assertIn('--interval-ms', got)
            self.assertIn('--worker', got)

    def test_normal_nsight_launcher_forwards_tegrastats(self):
        args = run.build_parser().parse_args([
            '--model', 'models/tinyllama', '--text-only', '--question', 'hello',
            '--benchmark', 'unused.nsys-rep', '--tegrastats',
            '--tegrastats-interval-ms', '140'
        ])
        with tempfile.TemporaryDirectory() as temp:
            args.benchmark = str(Path(temp) / 'capture.nsys-rep')
            got = []

            def fake_run(cmd, **_):
                got.extend(cmd)
                prefix = next(x[len('--output='):] for x in cmd if x.startswith('--output='))
                Path(prefix + '.nsys-rep').write_bytes(b'nonempty test report')
                return types.SimpleNamespace(returncode=0)

            with patch('benchmark_nsight.shutil.which', return_value='/usr/bin/nsys'), \
                 patch('benchmark_nsight.subprocess.run', side_effect=fake_run):
                self.assertEqual(benchmark_nsight.launch(args), 0)
            self.assertTrue(Path(args.benchmark).exists())
            self.assertIn('--tegrastats', got)
            self.assertIn('--tegrastats-interval-ms', got)
            self.assertIn('140', got)
            self.assertIn('--profile-worker', got)

    def test_suite_forwards_monitor_flags_to_three_commands(self):
        got = []

        def fake_run(cmd, **_):
            got.append(cmd)
            return types.SimpleNamespace(returncode=0)

        with patch.object(sys, 'argv', [
            'run_all.py', '--model', 'models/tinyllama',
            '--tegrastats', '--tegrastats-interval-ms', '200',
            '--output-dir', '/tmp/test_suite'
        ]), patch('run_all.subprocess.run', side_effect=fake_run):
            self.assertEqual(run_all.main(), 0)
        self.assertEqual(len(got), 3)
        for cmd in got:
            self.assertIn('--tegrastats', cmd)
            self.assertIn('--tegrastats-interval-ms', cmd)
            self.assertIn('200', cmd)
        self.assertIn('benchmark_latency.py', got[0][2])
        self.assertIn('benchmark_cpu.py', got[1][2])
        self.assertIn('benchmark_gpu.py', got[2][2])


if __name__ == '__main__':
    unittest.main()
