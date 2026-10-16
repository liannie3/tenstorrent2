import csv
import time
from datetime import datetime
from pathlib import Path
import torch

EVENTS = []  # (batch_size, iteration, label, t_ns)

## Stage boundaries recorded per iteration; durations are the gaps between them
STAGES = ('iter_start', 'decode', 'preprocess', 'encode', 'project')

def reset():
    EVENTS.clear()

def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()

def mark(batch_size, iteration, label, device):
    sync(device)
    EVENTS.append((batch_size, iteration, label, time.perf_counter_ns()))

## Per-iteration stage durations (ms) for one sub-test; incomplete iterations are dropped
def iteration_spans(batch_size):
    per_iter = {}
    for bs, iteration, label, t_ns in EVENTS:
        if bs == batch_size:
            per_iter.setdefault(iteration, {})[label] = t_ns

    spans = []
    for iteration in sorted(per_iter):
        marks = per_iter[iteration]
        if not all(label in marks for label in STAGES):
            continue
        spans.append({
            stage: (marks[stage] - marks[prev]) / 1e6
            for prev, stage in zip(STAGES, STAGES[1:])
        })
    return spans

## Rolls one sub-test's iterations up into mean stage latency and throughput
def summarize(batch_size, images):
    spans = iteration_spans(batch_size)
    if not spans:
        return None

    wall_ms = sum(sum(span.values()) for span in spans)
    return {
        'batch_size': batch_size,
        'images': images,
        'batches': len(spans),
        'mean_batch_ms': wall_ms / len(spans),
        'images_per_s': images / (wall_ms / 1000) if wall_ms else float('nan'),
        'stage_mean_ms': {
            stage: sum(span[stage] for span in spans) / len(spans)
            for stage in STAGES[1:]
        },
    }

def _stamped(results_dir, prefix):
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    return Path(results_dir) / f'{prefix}_{stamp}.csv'

def _writer(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(header)
    return f, w

## Raw marks for every sub-test in the sweep
def dump(path=None, results_dir='results'):
    path = Path(path) if path is not None else _stamped(results_dir, 'trace')
    f, w = _writer(path, ['batch_size', 'iteration', 'label', 't_ns'])
    with f:
        w.writerows(EVENTS)
    return path

## One row per sub-test
def dump_summary(summaries, path=None, results_dir='results'):
    path = Path(path) if path is not None else _stamped(results_dir, 'summary')
    header = ['batch_size', 'images', 'batches', 'mean_batch_ms', 'images_per_s']
    header += [f'{stage}_ms' for stage in STAGES[1:]]

    f, w = _writer(path, header)
    with f:
        for s in summaries:
            row = [s['batch_size'], s['images'], s['batches'],
                   round(s['mean_batch_ms'], 3), round(s['images_per_s'], 3)]
            row += [round(s['stage_mean_ms'][stage], 3) for stage in STAGES[1:]]
            w.writerow(row)
    return path
