#!/usr/bin/env python3
"""Run reproducible K-Means comparisons and draw Matplotlib PNG graphs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RAW = EXPERIMENTS / "raw"
GRAPHS = EXPERIMENTS / "graphs"
DATASETS = {
    "1m": ("higgs_1m_f32.bin", "higgs_1m_f32.metadata.json", 1_000_000),
    "11m": ("higgs_f32.bin", "higgs_f32.metadata.json", 11_000_000),
}
QUICK = [("sequential", 1, 1), ("openmp", 1, 4),
         ("mpi", 2, 1), ("hybrid", 2, 2)]
SCALING = ([("sequential", 1, 1)] +
           [("openmp", 1, t) for t in (1, 2, 4, 8, 16)] +
           [("mpi", p, 1) for p in (1, 2, 4, 8)] +
           [("hybrid", p, t) for p, t in ((2, 2), (2, 4), (4, 2), (4, 4))])
FIELDS = ["dataset", "rows", "clusters", "variant", "processes", "threads",
          "workers", "repeats", "iterations", "median_seconds", "speedup",
          "efficiency", "final_sse", "relative_sse_error",
          "max_centroid_error", "correct"]


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT,
                                       cwd=ROOT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def record_environment() -> None:
    data = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "gcc": command_output(["gcc", "--version"]).splitlines()[0],
        "mpicc": command_output(["mpicc", "--version"]).splitlines()[0],
        "mpirun": command_output(["mpirun", "--version"]).splitlines()[0],
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "location": str(ROOT),
        "dataset_bytes": {
            name: (ROOT / "dataset" / "processed" / files[0]).stat().st_size
            if (ROOT / "dataset" / "processed" / files[0]).exists() else None
            for name, files in DATASETS.items()
        },
    }
    (EXPERIMENTS / "environment.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_command(dataset: str, clusters: int, variant: str,
                  processes: int, threads: int, output: Path) -> list[str]:
    binary, metadata, _ = DATASETS[dataset]
    executable = str(ROOT / "build" / f"kmeans_{variant}")
    args = [executable, "--data", str(ROOT / "dataset" / "processed" / binary),
            "--metadata", str(ROOT / "dataset" / "processed" / metadata),
            "--clusters", str(clusters), "--seed", "42", "--output", str(output)]
    if variant in ("openmp", "hybrid"):
        args.extend(["--threads", str(threads)])
    if variant in ("mpi", "hybrid"):
        args = ["mpirun", "-np", str(processes), *args]
    return args


def run_case(dataset: str, clusters: int, variant: str, processes: int,
             threads: int, repeat: int, force: bool) -> dict:
    name = f"{dataset}_k{clusters}_{variant}_p{processes}_t{threads}_r{repeat}"
    path = RAW / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    result_path = RAW / f"{name}.result.json"
    command = build_command(dataset, clusters, variant, processes, threads, result_path)
    print(f"Running {name}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        print(completed.stdout[-4000:], file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_path.unlink()
    record = {"dataset": dataset, "clusters": clusters, "variant": variant,
              "processes": processes, "threads": threads, "repeat": repeat,
              "command": command, "result": result}
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"  iterations={result['iterations']} sse={result['final_sse']:.6f} "
          f"compute={result['computation_seconds']:.3f}s", flush=True)
    return record


def compare(reference: dict, candidate: dict) -> tuple[float, float, bool]:
    ref = reference["result"]
    other = candidate["result"]
    sse_error = abs(other["final_sse"] - ref["final_sse"]) / ref["final_sse"]
    centroid_error = max(
        abs(a - b)
        for row_a, row_b in zip(ref["final_centroids"], other["final_centroids"])
        for a, b in zip(row_a, row_b)
    )
    correct = (other["iterations"] == ref["iterations"] and
               sse_error <= 1e-5 and centroid_error <= 1e-4)
    return sse_error, centroid_error, correct


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["dataset"], record["clusters"], record["variant"],
               record["processes"], record["threads"])
        groups.setdefault(key, []).append(record)

    summary = []
    for (dataset, clusters, variant, processes, threads), runs in groups.items():
        reference_runs = groups[(dataset, clusters, "sequential", 1, 1)]
        baseline = statistics.median(
            run["result"]["computation_seconds"] for run in reference_runs)
        median = statistics.median(
            run["result"]["computation_seconds"] for run in runs)
        sse_error, centroid_error, correct = max(
            (compare(reference_runs[0], run) for run in runs),
            key=lambda values: (not values[2], values[0], values[1]))
        workers = processes * threads
        summary.append({
            "dataset": dataset, "rows": DATASETS[dataset][2], "clusters": clusters,
            "variant": variant, "processes": processes, "threads": threads,
            "workers": workers, "repeats": len(runs),
            "iterations": runs[0]["result"]["iterations"],
            "median_seconds": round(median, 6),
            "speedup": round(baseline / median, 6),
            "efficiency": round(baseline / median / workers, 6),
            "final_sse": runs[0]["result"]["final_sse"],
            "relative_sse_error": sse_error,
            "max_centroid_error": centroid_error,
            "correct": correct,
        })
    summary.sort(key=lambda row: (row["rows"], row["clusters"],
                                  row["variant"], row["workers"]))
    return summary


def save_csv(rows: list[dict]) -> None:
    with (EXPERIMENTS / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_saved_records() -> list[dict]:
    """Include completed runs from earlier invocations in the combined summary."""
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(RAW.glob("*.json"))]


def graph(rows: list[dict], dataset: str, clusters: int, metric: str) -> None:
    selected = [row for row in rows if row["dataset"] == dataset and
                row["clusters"] == clusters]
    if not selected:
        return
    selected.sort(key=lambda row: (row["variant"], row["workers"]))
    labels = [f"{row['variant']}  {row['processes']}x{row['threads']}" for row in selected]
    values = [float(row[metric]) for row in selected]
    colors = {"sequential": "#4b5563", "openmp": "#2563eb",
              "mpi": "#059669", "hybrid": "#db2777"}
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.45 * len(selected) + 1.6)))
    bars = ax.barh(labels, values, color=[colors[row["variant"]] for row in selected])
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlim(0, max(values) * 1.18 if max(values) else 1)
    ax.set_xlabel({"median_seconds": "Median computation time (seconds)",
                   "speedup": "Speedup vs sequential",
                   "efficiency": "Speedup / total workers"}[metric])
    ax.set_title(f"HIGGS {dataset} | K={clusters} | {metric.replace('_', ' ').title()}")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.01,
             f"Median of {selected[0]['repeats']} runs; loading and initialization excluded",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(GRAPHS / f"{dataset}_k{clusters}_{metric}.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "scaling"), default="quick")
    parser.add_argument("--dataset", choices=DATASETS, default="1m")
    parser.add_argument("--clusters", default="16", help="comma-separated K values")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    clusters = [int(item) for item in args.clusters.split(",")]
    if args.repeats < 1 or not clusters or any(k < 1 for k in clusters):
        parser.error("repeats and cluster counts must be positive")
    RAW.mkdir(parents=True, exist_ok=True)
    GRAPHS.mkdir(parents=True, exist_ok=True)
    record_environment()
    configurations = QUICK if args.suite == "quick" else SCALING
    for k in clusters:
        for variant, processes, threads in configurations:
            for repeat in range(1, args.repeats + 1):
                run_case(args.dataset, k, variant, processes,
                         threads, repeat, args.force)
        records = load_saved_records()
        rows = summarize(records)
        save_csv(rows)
        for dataset_name, cluster_count in sorted({
            (row["dataset"], row["clusters"]) for row in rows
        }):
            for metric in ("median_seconds", "speedup", "efficiency"):
                graph(rows, dataset_name, cluster_count, metric)
        if not all(row["correct"] for row in rows):
            print("Correctness check failed; inspect summary.csv", file=sys.stderr)
            return 1
    print(f"Saved {len(records)} runs to {RAW}; graphs to {GRAPHS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
