# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            hw_monitor.py
# CREATED:         11/17/2026
# REVISION:        1.0
# ======================================= #

"""TT hardware sampler -- temperature, power, current, voltage. TT's version of tegrastats.

benchmark_tests/hardware_perf.py launches it once per sub-test, like tegrastats on the
Orin. It also runs on its own in a second terminal, Ctrl+C to stop:

    python -m clipbench.tt.hw_monitor                  # every 100 ms
    python -m clipbench.tt.hw_monitor --interval 500

Reads the card's hwmon sensors from sysfs (tt-kmd) when they exist -- cheap and never
touches the device, so it can't disturb ttnn. Falls back to `tt-smi -s` snapshots
(~1 s each) if not. Writes one CSV row per sample, flushed so Ctrl+C loses nothing.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

## hwmon driver names that belong to a TT card
TT_HWMON_NAMES = ('blackhole', 'wormhole', 'tenstorrent', 'grayskull')

## sysfs prefix -> (unit, divisor). hwmon reports milli-units, power in micro-watts.
HWMON_UNITS = {'temp': ('c', 1e3), 'power': ('w', 1e6), 'curr': ('a', 1e3), 'in': ('v', 1e3)}


## Every TT hwmon dir, e.g. [/sys/class/hwmon/hwmon3]
def find_hwmon():
    found = []
    for d in sorted(Path('/sys/class/hwmon').glob('hwmon*')):
        try:
            name = (d / 'name').read_text().strip().lower()
        except OSError:
            continue
        if any(tt in name for tt in TT_HWMON_NAMES):
            found.append(d)
    return found


## {column: file} for every *_input sensor, named by its label when there is one
def hwmon_sensors(dirs):
    sensors = {}
    for card, d in enumerate(dirs):
        for f in sorted(d.glob('*_input')):
            prefix = f.name.split('_')[0].rstrip('0123456789')     # temp1_input -> temp
            if prefix not in HWMON_UNITS:
                continue
            label_file = d / f.name.replace('_input', '_label')
            label = label_file.read_text().strip() if label_file.exists() else f.name.replace('_input', '')
            unit, _ = HWMON_UNITS[prefix]
            sensors[f'card{card}_{label}_{unit}'.replace(' ', '_').lower()] = (f, prefix)
    return sensors


def read_hwmon(sensors):
    row = {}
    for col, (f, prefix) in sensors.items():
        try:
            row[col] = int(f.read_text().strip()) / HWMON_UNITS[prefix][1]
        except (OSError, ValueError):
            row[col] = ''
    return row


## tt-smi snapshot -> {cardN_<telemetry key>: value}. Run as a module so a broken
## console-script shebang (env moved from /mnt/data) doesn't matter.
def read_tt_smi():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'snap.json'
        subprocess.run([sys.executable, '-m', 'tt_smi', '-s', '-f', str(out)],
                       capture_output=True, check=True, timeout=30)
        snap = json.loads(out.read_text())

    row = {}
    for card, dev in enumerate(snap.get('device_info', [])):
        for key, value in (dev.get('telemetry') or {}).items():
            try:
                row[f'card{card}_{key}'] = float(value)
            except (TypeError, ValueError):
                row[f'card{card}_{key}'] = value
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sample TT card temperature/power to CSV')
    parser.add_argument('--interval', type=int, default=100, help='ms between samples (default 100)')
    parser.add_argument('--out', default=None, help='CSV path (default: today\'s benchmark folder)')
    parser.add_argument('--tt-smi', action='store_true', help='force tt-smi snapshots instead of sysfs')
    parser.add_argument('--quiet', action='store_true', help='no status lines (hardware_perf uses this)')
    args = parser.parse_args(argv)

    ## Same dated folder tt/run.py writes to
    if args.out:
        out = Path(args.out)
    else:
        today = date.today()
        out = (PROJECT_ROOT / 'analysis' / 'results' / 'tt'
               / f'benchmark_{today.month}-{today.day}-{today.year % 100}'
               / 'hardware_perf' / f'hw_{datetime.now():%H%M%S}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)

    sensors = {} if args.tt_smi else hwmon_sensors(find_hwmon())
    if sensors:
        read = lambda: read_hwmon(sensors)
        print(f'source: sysfs hwmon, {len(sensors)} sensors: {", ".join(sensors)}')
    else:
        read = read_tt_smi
        print('source: tt-smi snapshots (no TT hwmon in sysfs) -- about 1 sample/s at best')

    interval = args.interval / 1000
    if not args.quiet:
        print(f'writing {out}  (Ctrl+C to stop)')

    start = time.time()
    samples = 0
    writer = None
    with open(out, 'w', newline='') as f:
        try:
            while True:
                t = time.time()
                row = {'time': datetime.fromtimestamp(t).isoformat(timespec='milliseconds'),
                       'elapsed_s': round(t - start, 3), **read()}

                ## Columns fixed by the first sample
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row), extrasaction='ignore')
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                samples += 1

                if not args.quiet and samples % max(1, int(10 / interval)) == 1:     # status line about every 10 s
                    print('  ' + '  '.join(f'{k}={v}' for k, v in row.items() if k != 'time'), flush=True)

                time.sleep(max(0.0, interval - (time.time() - t)))
        except KeyboardInterrupt:
            pass

    print(f'  hw_monitor  : {samples} samples over {time.time() - start:.1f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
