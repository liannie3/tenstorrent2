# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            schema.py
# CREATED:         10/16/2026
# REVISION:        2.0
# ======================================= #

import csv
from pathlib import Path

COLUMNS = ('batch_size', 'iter', 'duration_ms')


def write_stage_csvs(spans, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    by_stage = {}
    for (batch_size, stage), durations in spans.items():
        by_stage.setdefault(stage, []).extend(
            (batch_size, index, duration)
            for index, duration in enumerate(durations)
        )

    paths = {}
    for stage, rows in by_stage.items():
        rows.sort()     ## batch size, then iteration
        path = directory / f'{stage}.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        paths[stage] = path

    return paths
