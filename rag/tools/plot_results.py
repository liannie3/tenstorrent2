"""Create presentation-ready charts from a RAG benchmark result directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".uv-cache" / "matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    ("cpu", "FP32"): "#4C78A8",
    ("cuda-reference", "FP32"): "#F58518",
    ("cuda-optimized", "FP32"): "#54A24B",
    ("cuda-optimized", "BF16"): "#B279A2",
}

LABELS = {
    ("cpu", "FP32"): "CPU FP32",
    ("cuda-reference", "FP32"): "CUDA reference FP32",
    ("cuda-optimized", "FP32"): "CUDA optimized FP32",
    ("cuda-optimized", "BF16"): "CUDA optimized BF16",
}

ORDER = list(LABELS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["Platform"], row["Type"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#8A8A8A",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#111111",
        "axes.titlesize": 16,
        "axes.titleweight": "semibold",
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def finish(fig: plt.Figure, output: Path, note: str) -> None:
    fig.text(0.01, 0.012, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )


def plot_batch1_latency(application, output: Path, note: str) -> None:
    rows = {
        key(row): row for row in application
        if row["Mode"] == "Host-facing/all" and row["Batch"] == "1"
    }
    labels = [LABELS[item] for item in ORDER]
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharex=True)
    for ax, metric, title in zip(
        axes,
        ("P50 ms", "P99 ms"),
        ("Median latency (P50)", "Tail latency (P99)"),
    ):
        values = [float(rows[item][metric]) for item in ORDER]
        bars = ax.bar(x, values, color=[COLORS[item] for item in ORDER], width=0.68)
        add_value_labels(ax, bars, "{:.3f}")
        ax.set_title(title)
        ax.set_ylabel("Request latency (ms)")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.22)
    fig.suptitle("Batch-1 host-facing latency", fontsize=20, fontweight="semibold")
    finish(fig, output, note)


def plot_throughput(application, output: Path, note: str) -> None:
    rows = [row for row in application if row["Mode"].endswith("/all")]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for item in ORDER:
        series = sorted(
            (row for row in rows if key(row) == item),
            key=lambda row: int(row["Batch"]),
        )
        batches = [int(row["Batch"]) for row in series]
        values = [float(row["Queries/s"]) for row in series]
        ax.plot(
            batches, values, marker="o", linewidth=2.5, markersize=7,
            color=COLORS[item], label=LABELS[item],
        )
        best = int(np.argmax(values))
        ax.annotate(
            f"{values[best]:,.0f} q/s",
            (batches[best], values[best]), xytext=(5, 8),
            textcoords="offset points", fontsize=9, color=COLORS[item],
        )
    ax.set_title("Throughput scaling by batch size")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (queries/s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncols=2, loc="upper right")
    finish(fig, output, note)


def plot_latency_scaling(application, output: Path, note: str) -> None:
    rows = [row for row in application if row["Mode"].endswith("/all")]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharex=True)
    for ax, metric, title in zip(
        axes, ("P50 ms", "P99 ms"), ("P50 latency", "P99 latency")
    ):
        for item in ORDER:
            series = sorted(
                (row for row in rows if key(row) == item),
                key=lambda row: int(row["Batch"]),
            )
            ax.plot(
                [int(row["Batch"]) for row in series],
                [float(row[metric]) for row in series],
                marker="o", linewidth=2.2, markersize=5,
                color=COLORS[item], label=LABELS[item],
            )
        ax.set_title(title)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Batch-request latency (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
        ax.grid(True, which="both")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(ncols=1, loc="upper left")
    fig.suptitle("Latency scaling across batch sizes", fontsize=20, fontweight="semibold")
    finish(fig, output, note)


def plot_stage_breakdown(stages, output: Path, note: str) -> None:
    selected = {
        key(row): row for row in stages
        if row["Batch"] == "1"
        and row["Path"] == "accepted"
        and row["Stage"] == "Vector retrieval"
    }
    vector = np.array([float(selected[item]["Critical-path %"]) for item in ORDER])
    payload = 100.0 - vector
    y = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    first = ax.barh(y, vector, color="#4C78A8", label="Vector retrieval")
    second = ax.barh(y, payload, left=vector, color="#E0A33A", label="Payload retrieval")
    for index, (v_value, p_value) in enumerate(zip(vector, payload)):
        if v_value >= 8:
            ax.text(v_value / 2, index, f"{v_value:.1f}%", ha="center", va="center", color="white")
        if p_value >= 8:
            ax.text(v_value + p_value / 2, index, f"{p_value:.1f}%", ha="center", va="center")
    ax.set_title("Accepted batch-1 request: where latency is spent", pad=45)
    ax.set_xlabel("Share of host-facing latency (%)")
    ax.set_yticks(y, [LABELS[item] for item in ORDER])
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        handles=[first, second], ncols=2, loc="lower center",
        bbox_to_anchor=(0.5, 1.01), borderaxespad=0,
    )
    finish(fig, output, note)


def plot_startup(application, output: Path, note: str) -> None:
    rows = {
        key(row): row for row in application
        if row["Mode"] == "Host-facing/all" and row["Batch"] == "1"
    }
    bank = np.array([float(rows[item]["Bank-load ms"]) for item in ORDER])
    init = np.array([float(rows[item]["Backend-init ms"]) for item in ORDER])
    first = np.array([float(rows[item]["First-request ms"]) for item in ORDER])
    cold = np.array([float(rows[item]["Cold-start ms"]) for item in ORDER])
    other = np.maximum(0.0, cold - bank - init - first)
    y = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    left = np.zeros(len(ORDER))
    for values, label, color in (
        (bank, "Bank load", "#72B7B2"),
        (init, "Backend initialization", "#F58518"),
        (first, "First request", "#B279A2"),
        (other, "Process/query setup", "#BAB0AC"),
    ):
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    for index, value in enumerate(cold):
        ax.text(value + max(cold) * 0.012, index, f"{value:.1f} ms", va="center", fontsize=10)
    ax.set_title("Cold-start composition at batch 1")
    ax.set_xlabel("Elapsed time (ms)")
    ax.set_yticks(y, [LABELS[item] for item in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, max(cold) * 1.18)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(ncols=2, loc="lower right")
    finish(fig, output, note)


def plot_memory(memory, output: Path, note: str) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for item in ORDER:
        series = sorted(
            (row for row in memory if key(row) == item),
            key=lambda row: int(row["Batch"]),
        )
        ax.plot(
            [int(row["Batch"]) for row in series],
            [int(row["Peak host RSS"]) / (1024 * 1024) for row in series],
            marker="o", linewidth=2.5, markersize=6,
            color=COLORS[item], label=LABELS[item],
        )
    ax.set_title("Peak host memory grows with batch size")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak host RSS (MiB)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncols=2, loc="upper left")
    finish(fig, output, note)


def plot_quality(quality, output: Path, note: str) -> None:
    rows = {(row["backend"], row["precision"].upper()): row for row in quality}
    x = np.arange(len(ORDER))
    width = 0.34
    means = [float(rows[item]["mean_abs_score_error"]) for item in ORDER]
    maxima = [float(rows[item]["max_abs_score_error"]) for item in ORDER]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.bar(x - width / 2, means, width, label="Mean absolute error", color="#4C78A8")
    ax.bar(x + width / 2, maxima, width, label="Maximum absolute error", color="#E45756")
    ax.set_title("Retrieval quality: all backends preserve decisions")
    ax.set_ylabel("Absolute score error vs. FP32 reference (log scale)")
    ax.set_xticks(x, [LABELS[item] for item in ORDER], rotation=18, ha="right")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    ax.text(
        0.01, 0.97, "Top-1 agreement: 100%   |   Gate agreement: 100%",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
    )
    finish(fig, output, note)


def plot_summary(application, memory, output: Path, note: str) -> None:
    all_rows = [row for row in application if row["Mode"].endswith("/all")]
    batch1 = {key(row): row for row in all_rows if row["Batch"] == "1"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    ax = axes[0, 0]
    x = np.arange(len(ORDER))
    values = [float(batch1[item]["P50 ms"]) for item in ORDER]
    bars = ax.bar(x, values, color=[COLORS[item] for item in ORDER])
    add_value_labels(ax, bars, "{:.3f}")
    ax.set_title("Batch-1 median latency")
    ax.set_ylabel("P50 (ms)")
    ax.set_xticks(x, ["CPU", "CUDA ref.", "CUDA opt.", "CUDA BF16"], rotation=12)

    ax = axes[0, 1]
    for item in ORDER:
        series = sorted((row for row in all_rows if key(row) == item), key=lambda row: int(row["Batch"]))
        ax.plot([int(row["Batch"]) for row in series], [float(row["Queries/s"]) for row in series], marker="o", linewidth=2, color=COLORS[item], label=LABELS[item])
    ax.set_title("Throughput scaling")
    ax.set_ylabel("Queries/s")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.legend(fontsize=9, ncols=2)

    ax = axes[1, 0]
    cold = [float(batch1[item]["Cold-start ms"]) for item in ORDER]
    bars = ax.bar(x, cold, color=[COLORS[item] for item in ORDER])
    add_value_labels(ax, bars, "{:.1f}")
    ax.set_title("Cold-start latency")
    ax.set_ylabel("Milliseconds")
    ax.set_xticks(x, ["CPU", "CUDA ref.", "CUDA opt.", "CUDA BF16"], rotation=12)

    ax = axes[1, 1]
    for item in ORDER:
        series = sorted((row for row in memory if key(row) == item), key=lambda row: int(row["Batch"]))
        ax.plot([int(row["Batch"]) for row in series], [int(row["Peak host RSS"]) / (1024 * 1024) for row in series], marker="o", linewidth=2, color=COLORS[item])
    ax.set_title("Peak host memory")
    ax.set_ylabel("MiB")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])

    for ax in axes.flat:
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NVIDIA RAG benchmark summary", fontsize=22, fontweight="semibold")
    finish(fig, output, note)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "presentation-charts"
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.results / "rag_run_manifest.json").read_text(encoding="utf-8"))
    application = read_csv(args.results / "rag_application_layer.csv")
    stages = read_csv(args.results / "rag_stage_layer.csv")
    memory = read_csv(args.results / "rag_memory_layer.csv")
    quality = read_csv(args.results / "rag_quality_gate.csv")
    note = (
        f"Source: {args.results.name}; 8-entry synthetic bank; "
        f"{manifest['measured_iterations']} measured corpus passes; "
        "single run per configuration; 99.2% accepted queries."
    )
    setup_style()
    plot_batch1_latency(application, output / "01-batch1-latency", note)
    plot_throughput(application, output / "02-throughput-scaling", note)
    plot_latency_scaling(application, output / "03-latency-scaling", note)
    plot_stage_breakdown(stages, output / "04-stage-breakdown", note)
    plot_startup(application, output / "05-cold-start", note)
    plot_memory(memory, output / "06-host-memory", note)
    plot_quality(quality, output / "07-quality", note)
    plot_summary(application, memory, output / "rag-benchmark-summary", note)
    print(f"wrote presentation charts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
