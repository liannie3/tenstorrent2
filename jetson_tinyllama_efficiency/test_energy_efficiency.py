"""Synthetic energy tests; no CUDA or Jetson needed."""
import unittest
from tegrastats_monitor import estimate_energy, parse_tegrastats, TegrastatsMonitor
from test_tegrastats_monitor import FakeTorch


class EnergyTests(unittest.TestCase):
    def test_constant_power_and_bounded_edges(self):
        points = [(i/10, 20.0) for i in range(1, 20)]
        r = estimate_energy(points, 0, 2, .1)
        self.assertEqual(r['status'], 'estimated')
        self.assertAlmostEqual(r['energy_j'], 40)
        self.assertAlmostEqual(r['observed_span_energy_j'], 36)
        self.assertAlmostEqual(r['coverage_percent'], 90)

    def test_linear_power_and_outside_samples(self):
        points = [(-1, 1000), (0, 10), (.1, 20), (.2, 30), (1, 1000)]
        r = estimate_energy(points, 0, .2, .1)
        self.assertAlmostEqual(r['energy_j'], 4)
        self.assertEqual(r['power_samples'], 3)

    def test_sparse_and_missing_data(self):
        for points in ([], [(0, 10)], [(0, 10), (1, 10)], [(.4, 10), (.5, 10)]):
            with self.subTest(points=points):
                r = estimate_energy(points, 0, 1, .1)
                self.assertNotIn('energy_j', r)

    def test_rail_group_and_clocks(self):
        m = parse_tegrastats('VDD_GPU_SOC 4000/3000 VDD_CPU_CV 2000/1900 '
                             'VIN_SYS_5V0 3000/2000 VDDQ_VDD2_1V8AO 1000/900 '
                             'CPU [25%@1000,off,40%@1200] GR3D_FREQ 50%@[900,1000]')
        self.assertEqual(m['agx_monitored_rails_power_w'], 9)
        self.assertEqual(m['cpu_core_2_clock_mhz'], 1200)
        self.assertNotIn('cpu_core_1_clock_mhz', m)
        self.assertEqual(m['gpu_gpc_1_clock_mhz'], 1000)
        self.assertNotIn('agx_monitored_rails_power_w', parse_tegrastats('VDD_GPU_SOC 3000/2000'))

    def test_efficiency_marker_and_temperature_scope(self):
        torch = FakeTorch()
        m = TegrastatsMonitor(100)
        m.samples = [(-1, {'gpu_temp_c': 99})] + [(i/10, {'vdd_in_power_w': 20, 'gpu_temp_c': 40+i}) for i in range(1, 20)]
        m.set_request_window(0, 2, 80, success=True)
        m.stop(torch)
        e = next(x for x in torch.cuda.nvtx.events if x.startswith('energy_efficiency'))
        self.assertIn('energy_j=40.000000', e)
        self.assertIn('joules_per_token=0.500000', e)
        self.assertIn('tokens_per_joule=2.000000', e)
        self.assertIn('energy_delay_j_s=80.000000', e)
        s = next(x for x in torch.cuda.nvtx.events if x.startswith('tegrastats_summary'))
        self.assertIn('gpu_temp_change_c=18.000', s)
        self.assertIn('gpu_temp_max_c=59.000', s)

    def test_failed_request_does_not_emit_efficiency(self):
        torch = FakeTorch()
        m = TegrastatsMonitor(100)
        m.samples = [(i/10, {'vdd_in_power_w': 20}) for i in range(1, 20)]
        m.set_request_window(0, 2, 80, success=False)
        m.stop(torch)
        self.assertIn('energy_efficiency status=request_failed', torch.cuda.nvtx.events)
        self.assertFalse(any('joules_per_token=' in x for x in torch.cuda.nvtx.events))


if __name__ == '__main__':
    unittest.main()
