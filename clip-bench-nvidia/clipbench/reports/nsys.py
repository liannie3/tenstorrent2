# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            nsys.py
# CREATED:         10/18/2026
# REVISION:        1.0
# ======================================= #

"""NVTX ranges in an nsys capture -> the same per-stage CSVs run_cpu.py writes.

    nsys profile --trace=cuda,nvtx,osrt --output results/jetson/sweep \
         python -m clipbench.run_nvidia --profile jetson

    python -m clipbench.reports.nsys results/jetson/sweep.nsys-rep
"""

import argparse
import csv
import re
import subprocess
from pathlib import Path

from .schema import write_stage_csvs

## Must stay in step with push() in clipbench/run_nvidia.py
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
        timings.append((batch_size, stage, index, float(row[duration_col]) / 1e6))

    return timings


def main(argv=None):
    parser = argparse.ArgumentParser(description='NVTX ranges in an nsys capture -> per-stage CSVs')
    parser.add_argument('capture', help='the .nsys-rep written by nsys profile')
    parser.add_argument('--report', default=NVTX_REPORT,
                        help=f'nsys stats report name (default: {NVTX_REPORT})')
    args = parser.parse_args(argv)

    csv_path = export_ranges(args.capture, args.report)
    print('nvtx export:', csv_path)

    timings = parse_ranges(csv_path)
    if not timings:
        raise SystemExit(f'no ranges matching {RANGE_PATTERN.pattern} - was --trace=nvtx set?')

    ## A subdirectory, not the capture's own folder: run_nvidia.py already wrote
    ## decode.csv, encode.csv and the rest there from its CUDA events, and those
    ## came from a run with no profiler attached. Writing on top of them would
    ## replace clean numbers with ones that carry CUPTI overhead.
    out_dir = Path(args.capture).parent / 'nsys'
    for stage, path in write_stage_csvs(timings, out_dir).items():
        print(f'{stage:>10}:', path)

    ## The ranges are host-side: nvtx_pushpop_trace records when the CPU thread
    ## pushed and popped, and a CUDA launch returns before the GPU runs it. So
    ## encode and project here are launch times, not GPU times -- the event
    ## numbers in the parent directory are the ones to quote. These rows are
    ## for seeing where host time goes and for lining up with the kernel
    ## timeline in the nsys GUI.
    print('\nthese rows are host-side range durations, not GPU time.')
    print('for GPU time per stage use the event CSVs written by run_nvidia.py.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
