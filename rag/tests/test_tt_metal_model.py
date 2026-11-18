"""Adversarial host tests for the fused TT-Metal reduction contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "tt_metal_model.py"
    spec = importlib.util.spec_from_file_location("tt_metal_model", path)
    model = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)

    for entries, workers in ((1, 64), (31, 8), (32, 8), (33, 8), (3400, 63)):
        ranges = model.partition_entries(entries, workers)
        assert ranges[0].begin == 0 and ranges[-1].end == entries
        assert all(a.end == b.begin for a, b in zip(ranges, ranges[1:]))
        assert max(r.end - r.begin for r in ranges) - min(
            r.end - r.begin for r in ranges
        ) <= 1

    C = model.Candidate
    assert model.better(C(0.5, 9), C(0.5, 2)).id == 2
    assert model.better(C(0.5, 2), C(100.0, 99, False)).id == 2
    assert model.reduce_candidates([C(0.7, 4)], 0.7).valid
    assert not model.reduce_candidates([C(0.699, 4)], 0.7).valid
    assert model.reduce_candidates([C(0.8, 7), C(0.8, 3)], 0.7).id == 3

    for args in ((0, 1), (1, 0)):
        try:
            model.partition_entries(*args)
            raise AssertionError("invalid partition input accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
