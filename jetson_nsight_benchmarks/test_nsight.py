"""Hardware-free wiring checks, not substitutes for a real Jetson capture."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from benchmark_stages import parser_for, profile_options, launch
from stage_observer import StageObserver


class NsightTests(unittest.TestCase):
    def args(self,mode='gpu'):
        return parser_for(mode).parse_args(['--model','models/test model','--text-only','--output','test.nsys-rep'])

    def observer(self,mode):
        self.syncs,self.labels,self.marks=[],[],[]
        @contextmanager
        def span(name):
            self.labels.append(name)
            yield
        torch=SimpleNamespace(cuda=SimpleNamespace(
            synchronize=lambda d:self.syncs.append(d),
            nvtx=SimpleNamespace(range=span,mark=self.marks.append)))
        return StageObserver(mode,'cuda:0',torch)

    def test_all_modes_capture_cuda_nvtx(self):
        for mode in ('latency','cpu','gpu'):
            opts=profile_options(self.args(mode))
            self.assertIn('--capture-range=cudaProfilerApi',opts)
            self.assertTrue(any(x.startswith('--trace=cuda,nvtx') for x in opts))
        self.assertIn('--sample=process-tree',profile_options(self.args('cpu')))
        self.assertIn('--sample=none',profile_options(self.args('gpu')))

    def test_explicit_cpu_fallback(self):
        args=self.args('cpu'); args.no_cpu_sampling=True
        self.assertIn('--sample=none',profile_options(args))
        self.assertIn('--cpuctxsw=none',profile_options(args))

    def test_latency_fences(self):
        ob=self.observer('latency')
        with ob.span('prefill'): pass
        self.assertEqual(len(self.syncs),2)
        self.assertIn('wall_ms=',self.marks[0])

    def test_cpu_process_time(self):
        ob=self.observer('cpu')
        with patch('stage_observer.time.perf_counter',side_effect=[1,1.2]), \
             patch('stage_observer.time.process_time',side_effect=[2,2.05]):
            with ob.span('tokenization'): pass
        self.assertEqual(self.syncs,[])
        self.assertIn('process_cpu_ms=50.000000',self.marks[0])
        self.assertIn('cpu_percent_one_core=25.000',self.marks[0])

    def test_gpu_has_no_stage_fences_or_fake_timings(self):
        ob=self.observer('gpu')
        with ob.span('prefill'): pass
        self.assertEqual(self.labels,['prefill'])
        self.assertEqual(self.syncs,[])
        self.assertEqual(self.marks,[])

    def test_launcher_preserves_inputs_and_report(self):
        for mode in ('latency','cpu','gpu'):
            with tempfile.TemporaryDirectory() as temp:
                a=self.args(mode); a.output=str(Path(temp)/'report.nsys-rep')
                def fake_run(cmd,check):
                    prefix=next(x.split('=',1)[1] for x in cmd if x.startswith('--output='))
                    Path(prefix+'.nsys-rep').write_bytes(b'mocked native report')
                    self.assertEqual(cmd[cmd.index('--model')+1],'models/test model')
                    self.assertEqual(cmd[cmd.index('--test')+1],mode)
                    self.assertIn('--worker',cmd)
                    return SimpleNamespace(returncode=0)
                with patch('benchmark_stages.shutil.which',return_value='/mock/nsys'), \
                     patch('benchmark_stages.subprocess.run',side_effect=fake_run):
                    self.assertEqual(launch(a),0)
                    self.assertEqual(Path(a.output).read_bytes(),b'mocked native report')
                    with self.assertRaises(FileExistsError): launch(a)

    def test_failed_capture_not_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.args(); a.output=str(Path(temp)/'failed.nsys-rep')
            with patch('benchmark_stages.shutil.which',return_value='/mock/nsys'), \
                 patch('benchmark_stages.subprocess.run',return_value=SimpleNamespace(returncode=2)):
                self.assertEqual(launch(a),2)
                self.assertFalse(Path(a.output).exists())


if __name__=='__main__':
    unittest.main()
