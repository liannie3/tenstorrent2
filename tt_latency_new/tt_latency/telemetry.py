"""Read-only Linux/TT telemetry. Acquisition windows share the latency clock."""
import csv
import json
import math
import os
from pathlib import Path
import re
import resource
from statistics import fmean
import subprocess
import threading
from time import perf_counter_ns, time_ns
from urllib.request import urlopen

# Explicit units avoid confusing electrical current with compute utilization.
EXPORT_METRICS = {
    'tt_asic_temperature_celsius': 'C', 'tt_board_temperature_celsius': 'C',
    'tt_ai_clock_mhz': 'MHz', 'tt_vcore_v': 'V',
    'tt_dispatch_core_efficiency_percent': '%', 'tt_dispatch_utilization_percent': '%',
    'tt_device_dram_total_megabytes': 'MiB', 'tt_device_dram_used_megabytes': 'MiB',
    'tt_last_update_timestamp': 'unix_s', 'tt_telemetry_collection_interval_seconds': 's',
    'tt_device_accessible': 'bool', 'tt_device_metrics_readable_percent': '%',
}
PROM_LINE = re.compile(r'^(\w+)(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def parse_prometheus(text):
    values = []
    for line in text.splitlines():
        match = PROM_LINE.match(line)
        if match and match[1] in EXPORT_METRICS:
            value = float(match[3])
            if math.isfinite(value):
                values.append((match[1], value, EXPORT_METRICS[match[1]], match[2] or '{}'))
    return values


def parse_snapshot(text):
    # Some TT-SMI versions write a progress prefix ahead of the JSON object.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(data, dict) and 'device_info' in data:
            return data
    raise ValueError('TT-SMI output contains no device_info JSON object')


def snapshot_metrics(data):
    values = []
    devices = data['device_info']
    items = devices.items() if isinstance(devices, dict) else enumerate(devices)
    for index, device in items:
        # Preserve identity and full telemetry even when an installed schema changes.
        labels = json.dumps({'snapshot_index': str(index), 'identity': {
            k: device[k] for k in ('board_info', 'device_id', 'pci_bus_id') if k in device}}, sort_keys=True)
        values.append(('raw_snapshot_telemetry', json.dumps(device), 'json', labels))
        for key, unit in [('asic_temperature', 'C'), ('power', 'W'), ('aiclk', 'MHz'), ('current', 'A'), ('board_power', 'W'), ('voltage', 'V'), ('fan_speed', 'RPM')]:
            value = device.get('telemetry', {}).get(key)
            # TT-SMI formats these documented fields as plain numeric strings.
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                values.append((key, '', unit, labels))
                continue
            if math.isfinite(number):
                values.append((key, number, unit, labels))
    return values


def discover_sensors(root=Path('/sys/class/hwmon')):
    sensors = []
    for directory in sorted(root.glob('hwmon*')):
        try:
            name = (directory / 'name').read_text().strip()
            device = (directory / 'device').resolve()
            vendor_paths = [device / 'vendor'] + [p / 'vendor' for p in device.parents]
            is_tt = any(p.exists() and p.read_text().strip().lower() == '0x1e52' for p in vendor_paths)
            if not is_tt and name not in ('coretemp', 'k10temp', 'zenpower', 'cpu_thermal'):
                continue
            for prefix, unit, scale in [('temp', 'C', 1000), ('power', 'W', 1000000),
                                        ('curr', 'A', 1000), ('in', 'V', 1000)]:
                for path in sorted(directory.glob(prefix + '*_input')):
                    label_path = path.with_name(path.name.replace('_input', '_label'))
                    label = label_path.read_text().strip() if label_path.exists() else path.stem
                    sensors.append((path, unit, scale, json.dumps({
                        'driver': name, 'sensor': label, 'device': str(device),
                        'scope': 'tt_device' if is_tt else 'host_cpu'})))
        except OSError:
            continue
    return sensors


class TelemetryMonitor:
    def __init__(self, interval=0.2, device_interval=2.0, tt_smi=False, metrics_url=None):
        self.interval, self.device_interval = interval, device_interval
        self.tt_smi, self.metrics_url = tt_smi, metrics_url
        self.rows, self.windows, self.threads = [], [], []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.sensors = discover_sensors()
        self.previous_cpu = None
        self.previous_process = None
        self.active_start = None

    def emit(self, source, metric, value, unit='', labels='{}', status='ok', start=None):
        end = perf_counter_ns()
        with self.lock:
            self.rows.append(dict(read_start_ns=start if start is not None else end,
                                  read_end_ns=end, unix_time_ns=time_ns(), source=source,
                                  metric=metric, value=value, unit=unit, labels=labels, status=status))

    def host_sample(self):
        start = perf_counter_ns()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = usage.ru_utime + usage.ru_stime
        if self.previous_process:
            previous_time, previous_cpu = self.previous_process
            percent = 100 * (cpu - previous_cpu) * 1e9 / (start - previous_time)
            self.emit('linux_process', 'cpu_percent_one_core', percent, '%', start=previous_time)
        self.previous_process = (start, cpu)
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        self.emit('/proc/self/statm', 'process_rss_bytes', rss, 'bytes', start=start)
        fields = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        total, idle = sum(fields), fields[3] + fields[4]
        if self.previous_cpu:
            previous_time, previous_total, previous_idle = self.previous_cpu
            delta = total - previous_total
            if delta > 0:
                percent = 100 * (1 - (idle - previous_idle) / delta)
                self.emit('/proc/stat', 'system_cpu_percent', percent, '%', start=previous_time)
        self.previous_cpu = (start, total, idle)
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        for name in ('MemTotal', 'MemAvailable'):
            self.emit('/proc/meminfo', name + '_bytes', memory[name], 'bytes', start=start)
        for path, unit, scale, labels in self.sensors:
            read_start = perf_counter_ns()
            try:
                self.emit(str(path), path.stem, float(path.read_text()) / scale,
                          unit, labels, start=read_start)
            except (OSError, ValueError) as error:
                self.emit(str(path), path.stem, '', unit, labels, str(error), read_start)

    def device_sample(self):
        if self.tt_smi:
            start = perf_counter_ns()
            try:
                result = subprocess.run(['tt-smi', '-s'], capture_output=True, text=True,
                                        timeout=5, check=True)
                values = snapshot_metrics(parse_snapshot(result.stdout))
                for metric, value, unit, labels in values:
                    self.emit('tt-smi -s', metric, value, unit, labels,
                              status='unavailable' if value == '' else 'ok', start=start)
            except (OSError, ValueError, subprocess.SubprocessError, TypeError, KeyError, AttributeError) as error:
                self.emit('tt-smi -s', 'device_sample', '', status=str(error)[:500], start=start)
        if self.metrics_url:
            start = perf_counter_ns()
            try:
                with urlopen(self.metrics_url, timeout=3) as response:
                    values = parse_prometheus(response.read().decode())
                if not values:
                    self.emit('tt-telemetry', 'device_sample', '', status='no_supported_metrics', start=start)
                for metric, value, unit, labels in values:
                    self.emit('tt-telemetry', metric, value, unit, labels, start=start)
            except (OSError, ValueError) as error:
                self.emit('tt-telemetry', 'device_sample', '', status=str(error)[:500], start=start)

    def loop(self, sample, interval):
        while not self.stop_event.is_set():
            start = perf_counter_ns()
            try:
                sample()
            except (OSError, ValueError, IndexError, KeyError) as error:
                self.emit('collector', sample.__name__, '', status=str(error), start=start)
            elapsed = (perf_counter_ns() - start) / 1e9
            self.stop_event.wait(max(0, interval - elapsed))

    def start(self):
        self.emit('collector', 'discovered_sensors', len(self.sensors), 'count')
        if not any('tt_device' in sensor[3] for sensor in self.sensors):
            self.emit('hwmon', 'tt_sensors', '', status='unavailable')
        if not self.metrics_url:
            self.emit('tt-telemetry', 'dispatch_and_device_memory', '', status='metrics_url_not_configured')
        for sample, interval in [(self.host_sample, self.interval), (self.device_sample, self.device_interval)]:
            thread = threading.Thread(target=self.loop, args=(sample, interval), daemon=True)
            self.threads.append(thread)
            thread.start()

    def begin_request(self, number):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.active_start = (number, perf_counter_ns(), usage.ru_utime + usage.ru_stime)

    def end_request(self):
        number, start, cpu = self.active_start
        end = perf_counter_ns()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.windows.append((number, start, end))
        return 100 * ((usage.ru_utime + usage.ru_stime) - cpu) * 1e9 / (end - start)

    def summary(self, number):
        _, start, end = next(window for window in self.windows if window[0] == number)
        groups = {}
        for row in self.rows:
            if (row['status'] == 'ok' and isinstance(row['value'], (int, float))
                    and start <= row['read_start_ns'] and row['read_end_ns'] <= end):
                key = (row['source'], row['metric'], row['unit'], row['labels'])
                groups.setdefault(key, []).append(row['value'])
        return [dict(source=source, metric=metric, unit=unit, labels=labels,
                     count=len(values), min=min(values), mean=fmean(values), max=max(values))
                for (source, metric, unit, labels), values in groups.items()]

    def finish(self, path):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()  # External reads have bounded timeouts, outside latency timers.
        with path.open('x', newline='', encoding='utf8') as stream:
            fields = ['run', 'overlapping_runs', 'read_start_ns', 'read_end_ns', 'unix_time_ns',
                      'source', 'metric', 'value', 'unit', 'labels', 'status']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in sorted(self.rows, key=lambda row: row['read_start_ns']):
                contained = [n for n, start, end in self.windows
                             if start <= row['read_start_ns'] and row['read_end_ns'] <= end]
                overlapping = [n for n, start, end in self.windows
                               if row['read_start_ns'] <= end and row['read_end_ns'] >= start]
                writer.writerow(dict(run=contained[0] if contained else '',
                                     overlapping_runs=json.dumps(overlapping), **row))
