# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            data.py
# CREATED:         10/16/2026
# REVISION:        1.0
# ======================================= #

"""The one results shape every target and every report reader produces.

NVTX ranges read out of an nsys capture, perf_counter spans measured on CPU,
and whatever a future accelerator reports all normalise to these columns. That
is what keeps cross-platform comparison a concatenation instead of a
translation layer.
"""

import csv
from pathlib import Path

COLUMNS = (
    'platform', 'device', 'precision', 'batch_size',
    'stage', 'n', 'mean_ms', 'median_ms', 'p95_ms', 'images_per_s',
)

## Pipeline order for display. 'subtest' is the envelope around a whole batch
## size, 'iter' the envelope around one batch; the rest sum into 'iter'.
STAGE_ORDER = ('subtest', 'iter', 'decode', 'preprocess', 'encode', 'project')


def make_row(**values):
    unknown = set(values) - set(COLUMNS)
    if unknown:
        raise ValueError(f'Unknown result columns: {sorted(unknown)}')
    return {column: values.get(column, '') for column in COLUMNS}


def stage_sort_key(row):
    stage = row['stage']
    order = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    return (row['batch_size'], order, stage)


## Nearest-rank rather than interpolated, so a short run reports a measurement
## that actually happened instead of a number between two that did
def percentile(values, fraction):
    if not values:
        return float('nan')
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(rows):
    if not rows:
        print('  (no rows)')
        return
    header = f"{'bs':>4}  {'stage':<11}{'n':>5}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}{'img/s':>10}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f"{row['batch_size']:>4}  {row['stage']:<11}{row['n']:>5}"
              f"{row['mean_ms']:>10}{row['median_ms']:>10}{row['p95_ms']:>10}"
              f"{row['images_per_s']:>10}")
