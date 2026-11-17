"""Summarize only device operations between measured request signposts."""
from collections import defaultdict
import csv
import math

from metrics import write_csv


def summarize(capture, destination, requests):
    reports = list(capture.rglob("ops_perf_results*.csv"))
    traces = [p for p in capture.rglob("*.tracy") if p.stat().st_size]
    if len(reports) != 1 or not traces:
        raise RuntimeError(f"Expected one op CSV and a Tracy trace in {capture}. Found {reports}, {traces}")
    totals = defaultdict(list)
    active = False
    count = 0
    with reports[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            op = row["OP CODE"]
            if row["OP TYPE"].lower() == "signpost":
                if op == "measured_begin":
                    if active:
                        raise ValueError("Overlapping request markers")
                    active = True
                    count += 1
                elif op == "measured_end":
                    if not active:
                        raise ValueError("Unpaired end marker")
                    active = False
                continue
            if active:
                raw = row.get("DEVICE KERNEL DURATION [ns]", "").strip()
                try:
                    duration = float(raw)
                except ValueError:
                    continue
                if math.isfinite(duration) and duration > 0:
                    totals[op].append(duration/1e6)
    if active or count != requests or not totals:
        raise RuntimeError("Missing measured request markers or device durations. Inspect the raw Tracy report.")
    rows = [dict(op=op, calls=len(v), sum_kernel_ms=sum(v), mean_kernel_ms=sum(v)/len(v),
                 max_kernel_ms=max(v)) for op,v in totals.items()]
    write_csv(destination, sorted(rows, key=lambda row: -row["sum_kernel_ms"]))
