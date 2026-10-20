# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            nsys.py
# CREATED:         10/18/2026
# REVISION:        1.0
# ======================================= #

"""NVTX ranges in an nsys capture -> an nvtx_ms column in the stage CSVs.

run_nvidia.py runs this itself. To redo it on an existing capture:

    python -m clipbench.reports.nsys results/jetson/sweep.nsys-rep
"""

import argparse
import csv
import re
import subprocess
from pathlib import Path

from .schema_nvidia import merge_nvtx_column

## Must stay in step with push() in clipbench/run_nvidia.py
RANGE_PATTERN = re.compile(r'^(?P<stage>\w+)/bs=(?P<batch_size>\d+)$')

## One row per range instance, not per range name. Report names vary between
## nsys versions -- `nsys stats --help-reports` lists what yours has.
NVTX_REPORT = 'nvtx_pushpop_trace'

## nsys labels the column with its unit, e.g. 'Duration (ns)'
PER_MS = {'ns': 1e6, 'us': 1e3, 'µs': 1e3, 'ms': 1.0, 's': 1e-3}


def export_ranges(rep_path, report=NVTX_REPORT):
    rep_path = Path(rep_path)
    if not rep_path.is_file():
        raise FileNotFoundError(f'no capture at {rep_path}')

    try:
        result = subprocess.run(
            ['nsys', 'stats', '--report', report, '--format', 'csv', str(rep_path)],
            capture_output=True, text=True,
        )
    except FileNotFoundError:   # Checks if nsys exists
        raise RuntimeError('nsys not on PATH - run this on the Orin, or install Nsight Systems')

    if result.returncode != 0:  # Checks if nsys ran but failed
        raise RuntimeError(f'nsys stats --report {report} failed:\n{result.stderr.strip()}')

    csv_path = rep_path.with_name(f'{rep_path.stem}_nvtx.csv')
    csv_path.write_text(result.stdout)
    return csv_path


def parse_ranges(csv_path):
    lines = Path(csv_path).read_text().splitlines()

    ## nsys prints progress lines and a report title before the CSV itself
    header = next((i for i, line in enumerate(lines)
                   if 'Duration' in line and 'Name' in line), None)
    if header is None:
        raise RuntimeError(f'no CSV header in {csv_path} - wrong report?')

    reader = csv.DictReader(lines[header:])
    duration_col = next(c for c in reader.fieldnames if c.startswith('Duration'))

    ## Read the unit off the header rather than assuming ns
    unit = re.search(r'\(([^)]+)\)', duration_col)
    unit = unit.group(1).strip().lower() if unit else 'ns'
    if unit not in PER_MS:
        raise RuntimeError(f'unexpected duration unit {unit!r} in column {duration_col!r}')
    per_ms = PER_MS[unit]

    ## Same (batch_size, stage, iter, duration_ms) rows a runner buffers.
    ## The range name has no iteration index in it on purpose, so the index is
    ## the position within its (batch size, stage) group -- which works because
    ## nsys lists push/pop ranges in start order.
    counts = {}
    timings = []
    for row in reader:
        match = RANGE_PATTERN.match(row['Name'])
        if not match:
            continue
        stage = match['stage']
        batch_size = int(match['batch_size'])
        index = counts.get((batch_size, stage), 0)
        counts[(batch_size, stage)] = index + 1
        timings.append((batch_size, stage, index, float(row[duration_col]) / per_ms))

    return timings


def main(argv=None):
    parser = argparse.ArgumentParser(description='NVTX ranges in an nsys capture -> nvtx_ms column')
    parser.add_argument('capture', help='the .nsys-rep written by nsys profile')
    parser.add_argument('--report', default=NVTX_REPORT,
                        help=f'nsys stats report name (default: {NVTX_REPORT})')
    parser.add_argument('--csv-dir', default=None,
                        help="stage CSVs to merge into (default: the capture's own directory)")
    args = parser.parse_args(argv)

    csv_path = export_ranges(args.capture, args.report)
    print('nvtx export:', csv_path)

    timings = parse_ranges(csv_path)
    if not timings:
        raise SystemExit(f'no ranges matching {RANGE_PATTERN.pattern} - was --trace=nvtx set?')

    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.capture).parent
    merged = merge_nvtx_column(timings, csv_dir)
    if not merged:
        raise SystemExit(f'no stage CSVs in {csv_dir} to merge into')

    for stage, path in sorted(merged.items()):
        print(f'{stage:>10}:', path)

    ## Ranges are host-side, so encode and project are launch times, not GPU
    ## times. duration_ms beside them is the number to quote.
    print('\nnvtx_ms is host-side range duration, not GPU time.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
