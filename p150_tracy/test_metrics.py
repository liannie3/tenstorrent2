import csv
from pathlib import Path
import tempfile
import unittest
from metrics import Observer, energy
from report import summarize


class Checks(unittest.TestCase):
    def test_token_intervals(self):
        obs = Observer()
        obs.tokens = [2, 3, 5]
        metrics = obs.timing(0, 6)
        self.assertEqual(metrics["ttft_ms"], 2000)
        self.assertEqual(metrics["tpot_ms"], 1500)
        obs.tokens = [2]
        self.assertIsNone(obs.timing(0, 3)["decode_tokens_s"])

    def test_boundaries_and_missing_power(self):
        rows = [dict(time_s=t, power_w=2*t) for t in range(5)]
        self.assertAlmostEqual(energy(rows, 0.5, 3.5, 1)[0], 12)
        rows[2]["power_w"] = None
        self.assertIsNone(energy(rows, 0.5, 3.5, 1)[0])
        self.assertIsNone(energy(rows, 0.1, 0.9, 1)[0])

    def test_tracy_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"capture.tracy").write_bytes(b"fixture")
            with (root/"ops_perf_results_x.csv").open("w") as f:
                f.write("OP CODE,OP TYPE,DEVICE KERNEL DURATION [ns]\nwarm,device,999999\n"
                        "measured_begin,signpost,\nmatmul,device,2000000\nmeasured_end,signpost,\n")
            summarize(root, root/"out.csv", 1)
            with (root/"out.csv").open() as f:
                result = list(csv.DictReader(f))
            self.assertEqual([(r["op"],float(r["sum_kernel_ms"])) for r in result],[("matmul",2)])
            with self.assertRaises(RuntimeError):
                summarize(root, root/"out.csv", 2)
