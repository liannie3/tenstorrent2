# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            nsys.py
# CREATED:         10/18/2026
# REVISION:        1.0
# ======================================= #

"""NVTX ranges in an nsys capture -> the same per-stage CSVs the CPU target writes.

    nsys profile --trace=cuda,nvtx,osrt --output results/jetson/sweep \
         python -m clipbench.runner --profile jetson

    python -m clipbench.reports.nsys results/jetson/sweep.nsys-rep
"""

import argparse
import csv
import re
import subprocess
from pathlib import Path

from .schema import write_stage_csvs

## Must stay in step with range_name() in clipbench/targets/nvidia.py
RANGE_PATTERN = re.compile(r'^(?P<stage>\w+)/bs=(?P<batch_size>\d+)$')

## One row per range instance, not per range name. Report names vary between
## nsys versions -- `nsys stats --help-reports` lists what yours has.
NVTX_REPORT = 'nvtx_pushpop_trace'


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

    spans = {}
    for row in reader:
        match = RANGE_PATTERN.match(row['Name'])
        if not match:
            continue
        key = (int(match['batch_size']), match['stage'])
        spans.setdefault(key, []).append(float(row[duration_col]) / 1e6)

    return spans


def main(argv=None):
    parser = argparse.ArgumentParser(description='NVTX ranges in an nsys capture -> per-stage CSVs')
    parser.add_argument('capture', help='the .nsys-rep written by nsys profile')
    parser.add_argument('--report', default=NVTX_REPORT,
                        help=f'nsys stats report name (default: {NVTX_REPORT})')
    args = parser.parse_args(argv)

    csv_path = export_ranges(args.capture, args.report)
    print('nvtx export:', csv_path)

    spans = parse_ranges(csv_path)
    if not spans:
        raise SystemExit(f'no ranges matching {RANGE_PATTERN.pattern} - was --trace=nvtx set?')

    for stage, path in write_stage_csvs(spans, Path(args.capture).parent).items():
        print(f'{stage:>10}:', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
