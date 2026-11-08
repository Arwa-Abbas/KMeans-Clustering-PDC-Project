#!/usr/bin/env python3
"""Run reproducible K-Means scaling experiments and draw Matplotlib PNG graphs.

Suites:
  quick    sequential, OpenMP 4 threads, MPI 2 processes, hybrid 2x2
  scaling  strong scaling: a fixed dataset, 1-16 workers
  weak     weak scaling: a fixed number of rows per worker, fixed iteration count
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bump when a code change makes earlier measurements incomparable. Version 2: C++17 code
# base, one shared assignment kernel, one MPI_Allreduce per iteration, unpinned MPI ranks.
VERSION = 2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RAW = EXPERIMENTS / "raw"
GRAPHS = EXPERIMENTS / "graphs"
DATASETS = {
    "1m": ("higgs_1m_f32.bin", "higgs_1m_f32.metadata.json", 1_000_000),
    "11m": ("higgs_f32.bin", "higgs_f32.metadata.json", 11_000_000),
}
QUICK = [("sequential", 1, 1), ("openmp", 1, 4), ("mpi", 2, 1), ("hybrid", 2, 2)]
SCALING = ([("sequential", 1, 1)] +
           [("openmp", 1, t) for t in (1, 2, 4, 8, 16)] +
           [("mpi", p, 1) for p in (1, 2, 4, 8)] +
           [("hybrid", p, t) for p, t in ((2, 2), (2, 4), (4, 2), (4, 4))])
WEAK_WORKERS = (1, 2, 4, 8)
WEAK = ([("sequential", 1, 1)] +
        [("openmp", 1, w) for w in WEAK_WORKERS] +
        [("mpi", w, 1) for w in WEAK_WORKERS] +
        [("hybrid", p, t) for p, t in ((2, 2), (2, 4), (4, 2))])
# Open MPI pins each rank to one core when it starts 2 or fewer ranks, which would squeeze
# all of a hybrid rank's OpenMP threads onto that core. Leave placement to the OS for every
# variant instead, so all runs are scheduled the same way.
MPIRUN = ["mpirun", "--bind-to", "none"]

STRONG_FIELDS = ["dataset", "rows", "clusters", "variant", "processes", "threads",
                 "workers", "repeats", "iterations", "median_seconds", "speedup",
                 "efficiency", "final_sse", "relative_sse_error",
                 "max_centroid_error", "correct"]
WEAK_FIELDS = ["clusters", "rows", "rows_per_worker", "variant", "processes", "threads",
               "workers", "repeats", "iterations", "median_seconds",
               "seconds_per_iteration", "weak_efficiency", "final_sse",
               "relative_sse_error", "max_centroid_error", "correct"]

# Categorical slots 1-3 (validated all-pairs), plus neutral ink for the baseline.
COLORS = {"sequential": "#52514e", "openmp": "#2a78d6", "mpi": "#eb6834", "hybrid": "#1baf7a"}
MARKERS = {"sequential": "s", "openmp": "o", "mpi": "D", "hybrid": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


@dataclass(frozen=True)
class Case:
    suite: str            # "strong" or "weak"
    dataset: str          # which binary is read
    rows: int             # rows clustered: the whole binary or a prefix of it
    clusters: int
    variant: str
    processes: int
    threads: int
    max_iterations: int   # 0: program default with the 0.1% stopping rule
    repeat: int

    @property
    def name(self) -> str:
        layout = f"{self.variant}_p{self.processes}_t{self.threads}_r{self.repeat}"
        if self.suite == "strong":
            return f"{self.dataset}_k{self.clusters}_{layout}"
        return f"weak_n{self.rows}_i{self.max_iterations}_k{self.clusters}_{layout}"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT,
                                       cwd=ROOT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def first_line(command: list[str]) -> str:
    return (command_output(command).splitlines() or ["unavailable"])[0]


def record_environment() -> None:
    model = "unavailable"
    for line in command_output(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            model = line.split(":", 1)[1].strip()
    cores = {line for line in command_output(["lscpu", "-p=Core,Socket"]).splitlines()
             if line and not line.startswith("#")}
    data = {
        "results_version": VERSION,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": model,
        "physical_cores": len(cores) or None,
        "logical_cpus": os.cpu_count(),
        "compiler": first_line(["g++", "--version"]),
        "mpicxx": first_line(["mpicxx", "--version"]),
        "mpirun": first_line(["mpirun", "--version"]),
        "mpi_placement": " ".join(MPIRUN),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "unset"),
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


def build_command(case: Case, output: Path) -> list[str]:
    binary, metadata, total_rows = DATASETS[case.dataset]
    args = [str(ROOT / "build" / f"kmeans_{case.variant}"),
            "--data", str(ROOT / "dataset" / "processed" / binary),
            "--metadata", str(ROOT / "dataset" / "processed" / metadata),
            "--clusters", str(case.clusters), "--seed", "42", "--output", str(output)]
    if case.rows != total_rows:
        args += ["--rows", str(case.rows)]
    if case.max_iterations:
        # Tolerance 0 never stops early, so every weak-scaling size runs the same iterations.
        args += ["--max-iterations", str(case.max_iterations), "--tolerance", "0"]
    if case.variant in ("openmp", "hybrid"):
        args += ["--threads", str(case.threads)]
    if case.variant in ("mpi", "hybrid"):
        args = [*MPIRUN, "-np", str(case.processes), *args]
    return args


def run_case(case: Case, force: bool) -> dict:
    path = RAW / f"{case.name}.json"
    if path.exists() and not force:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("version") == VERSION:
            return record

    result_path = RAW / f"{case.name}.result.json"
    command = build_command(case, result_path)
    print(f"Running {case.name}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        print(completed.stdout[-4000:], file=sys.stderr)
        raise RuntimeError(f"{case.name} failed with exit code {completed.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_path.unlink()
    record = {"version": VERSION, **asdict(case), "command": command, "result": result}
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"  iterations={result['iterations']} sse={result['final_sse']:.6f} "
          f"compute={result['computation_seconds']:.3f}s", flush=True)
    return record


def plan(args: argparse.Namespace) -> list[Case]:
    total_rows = DATASETS[args.dataset][2]
    cases = []
    for k in args.clusters:
        if args.suite == "weak":
            for variant, processes, threads in WEAK:
                # The sequential reference runs at every size used, for correctness checks.
                sizes = WEAK_WORKERS if variant == "sequential" else (processes * threads,)
                for workers in sizes:
                    rows = args.weak_rows_per_worker * workers
                    if rows > total_rows:
                        raise SystemExit(f"weak scaling needs {rows} rows; "
                                         f"{args.dataset} has {total_rows}")
                    cases += [Case("weak", args.dataset, rows, k, variant, processes, threads,
                                   args.weak_iterations, r)
                              for r in range(1, args.repeats + 1)]
        else:
            configurations = QUICK if args.suite == "quick" else SCALING
            cases += [Case("strong", args.dataset, total_rows, k, variant, processes,
                           threads, 0, r)
                      for variant, processes, threads in configurations
                      for r in range(1, args.repeats + 1)]
    return cases


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


def median_seconds(runs: list[dict]) -> float:
    return statistics.median(run["result"]["computation_seconds"] for run in runs)


def worst_comparison(reference: list[dict], runs: list[dict]) -> tuple[float, float, bool]:
    return max((compare(reference[0], run) for run in runs),
               key=lambda values: (not values[2], values[0], values[1]))


def group(records: list[dict], suite: str, key) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        if record["suite"] == suite:
            groups.setdefault(key(record), []).append(record)
    return groups


def summarize_strong(records: list[dict]) -> list[dict]:
    groups = group(records, "strong", lambda r: (r["dataset"], r["clusters"], r["variant"],
                                                  r["processes"], r["threads"]))
    summary = []
    for (dataset, clusters, variant, processes, threads), runs in groups.items():
        reference = groups.get((dataset, clusters, "sequential", 1, 1))
        if not reference:
            continue
        baseline = median_seconds(reference)
        median = median_seconds(runs)
        sse_error, centroid_error, correct = worst_comparison(reference, runs)
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
    summary.sort(key=lambda row: (row["rows"], row["clusters"], row["variant"], row["workers"]))
    return summary


def summarize_weak(records: list[dict]) -> list[dict]:
    groups = group(records, "weak", lambda r: (r["clusters"], r["max_iterations"], r["rows"],
                                                r["variant"], r["processes"], r["threads"]))
    summary = []
    for (clusters, iterations, rows, variant, processes, threads), runs in groups.items():
        workers = processes * threads
        per_worker = rows // workers
        # T1: one worker on one worker's share; ideal weak scaling keeps time constant.
        baseline_runs = groups.get((clusters, iterations, per_worker, "sequential", 1, 1))
        reference = groups.get((clusters, iterations, rows, "sequential", 1, 1))
        if not baseline_runs or not reference:
            continue
        median = median_seconds(runs)
        sse_error, centroid_error, correct = worst_comparison(reference, runs)
        summary.append({
            "clusters": clusters, "rows": rows, "rows_per_worker": per_worker,
            "variant": variant, "processes": processes, "threads": threads,
            "workers": workers, "repeats": len(runs),
            "iterations": runs[0]["result"]["iterations"],
            "median_seconds": round(median, 6),
            "seconds_per_iteration": round(median / runs[0]["result"]["iterations"], 6),
            "weak_efficiency": round(median_seconds(baseline_runs) / median, 6),
            "final_sse": runs[0]["result"]["final_sse"],
            "relative_sse_error": sse_error,
            "max_centroid_error": centroid_error,
            "correct": correct,
        })
    summary.sort(key=lambda row: (row["clusters"], row["variant"], row["workers"], row["rows"]))
    return summary


def save_csv(name: str, fields: list[str], rows: list[dict]) -> None:
    with (EXPERIMENTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_saved_records() -> list[dict]:
    """Completed runs from earlier invocations; older result versions are ignored."""
    records = (json.loads(path.read_text(encoding="utf-8")) for path in sorted(RAW.glob("*.json")))
    return [record for record in records if record.get("version") == VERSION]


def style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def footnote(fig, repeats: int, extra: str = "") -> None:
    text = f"Median of {repeats} runs; loading and initialization excluded{extra}"
    fig.text(0.5, 0.01, text, ha="center", fontsize=9, color=MUTED)


def bar_graph(rows: list[dict], dataset: str, clusters: int, metric: str) -> None:
    selected = sorted((row for row in rows if row["dataset"] == dataset and
                       row["clusters"] == clusters),
                      key=lambda row: (row["variant"], row["workers"]))
    if not selected:
        return
    labels = [f"{row['variant']}  {row['processes']}x{row['threads']}" for row in selected]
    values = [float(row[metric]) for row in selected]
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.45 * len(selected) + 1.6)))
    bars = ax.barh(labels, values, height=0.7, color=[COLORS[row["variant"]] for row in selected],
                   edgecolor="white", linewidth=2)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.18 if max(values) else 1)
    ax.set_xlabel({"median_seconds": "Median computation time (seconds)",
                   "speedup": "Speedup vs sequential",
                   "efficiency": "Speedup / total workers"}[metric], color=INK)
    ax.set_title(f"HIGGS {dataset} | K={clusters} | {metric.replace('_', ' ').title()}", color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    footnote(fig, selected[0]["repeats"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(GRAPHS / f"{dataset}_k{clusters}_{metric}.png", dpi=180)
    plt.close(fig)


def scaling_lines(ax, rows: list[dict], metric: str) -> None:
    """OpenMP and MPI as lines over worker count; hybrid layouts as labelled points."""
    for variant in ("openmp", "mpi"):
        points = sorted((row["workers"], float(row[metric])) for row in rows
                        if row["variant"] == variant)
        if points:
            ax.plot(*zip(*points), color=COLORS[variant], linewidth=2, marker=MARKERS[variant],
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    label="OpenMP threads" if variant == "openmp" else "MPI processes")
    hybrid = [row for row in rows if row["variant"] == "hybrid"]
    if hybrid:
        ax.scatter([row["workers"] for row in hybrid], [float(row[metric]) for row in hybrid],
                   s=80, color=COLORS["hybrid"], marker=MARKERS["hybrid"], edgecolors="white",
                   linewidths=1.5, zorder=3, label="Hybrid processes x threads")
        for row in hybrid:
            ax.annotate(f"{row['processes']}x{row['threads']}", (row["workers"], float(row[metric])),
                        textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK)


def worker_axis(ax, workers: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers])
    ax.set_xlabel("Total workers (processes x threads)", color=INK)


def strong_graph(rows: list[dict], dataset: str, clusters: int) -> None:
    selected = [row for row in rows if row["dataset"] == dataset and row["clusters"] == clusters
                and row["variant"] != "sequential"]
    if not selected:
        return
    workers = sorted({row["workers"] for row in selected})
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(workers, workers, color=MUTED, linewidth=1.2, linestyle="--", label="Ideal (linear)")
    scaling_lines(ax, selected, "speedup")
    worker_axis(ax, workers)
    ax.set_yscale("log", base=2)
    ax.set_yticks(workers)
    ax.set_yticklabels([str(w) for w in workers])
    ax.set_ylabel("Speedup vs sequential", color=INK)
    ax.set_title(f"Strong scaling | HIGGS {dataset} | K={clusters}", color=INK)
    style(ax)
    ax.legend(frameon=False, labelcolor=INK)
    footnote(fig, selected[0]["repeats"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(GRAPHS / f"{dataset}_k{clusters}_scaling.png", dpi=180)
    plt.close(fig)


def weak_graph(rows: list[dict], clusters: int) -> None:
    selected = [row for row in rows if row["clusters"] == clusters and row["variant"] != "sequential"]
    if not selected:
        return
    workers = sorted({row["workers"] for row in selected})
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle="--", label="Ideal (constant time)")
    scaling_lines(ax, selected, "weak_efficiency")
    worker_axis(ax, workers)
    ax.set_ylim(0, max(1.15, max(float(row["weak_efficiency"]) for row in selected) * 1.1))
    ax.set_ylabel("Weak-scaling efficiency  T(1 worker) / T(p workers)", color=INK)
    per_worker = selected[0]["rows_per_worker"]
    ax.set_title(f"Weak scaling | {per_worker:,} rows per worker | K={clusters}", color=INK)
    style(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    footnote(fig, selected[0]["repeats"], f"; {selected[0]['iterations']} iterations per run")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(GRAPHS / f"weak_k{clusters}_efficiency.png", dpi=180)
    plt.close(fig)


def report(records: list[dict]) -> bool:
    strong = summarize_strong(records)
    weak = summarize_weak(records)
    save_csv("summary.csv", STRONG_FIELDS, strong)
    save_csv("weak_summary.csv", WEAK_FIELDS, weak)
    for dataset, clusters in sorted({(row["dataset"], row["clusters"]) for row in strong}):
        for metric in ("median_seconds", "speedup", "efficiency"):
            bar_graph(strong, dataset, clusters, metric)
        strong_graph(strong, dataset, clusters)
    for clusters in sorted({row["clusters"] for row in weak}):
        weak_graph(weak, clusters)
    return all(row["correct"] for row in strong + weak)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("quick", "scaling", "weak"), default="quick")
    parser.add_argument("--dataset", choices=DATASETS,
                        help="binary to read (default: 1m for quick/scaling, 11m for weak)")
    parser.add_argument("--clusters", default="16", help="comma-separated K values")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--weak-rows-per-worker", type=int, default=1_375_000,
                        help="weak-scaling rows per worker (default: 8 workers use all 11M)")
    parser.add_argument("--weak-iterations", type=int, default=20,
                        help="fixed iteration count for weak-scaling runs")
    parser.add_argument("--force", action="store_true", help="rerun saved repetitions")
    args = parser.parse_args()
    args.dataset = args.dataset or ("11m" if args.suite == "weak" else "1m")
    args.clusters = [int(item) for item in args.clusters.split(",")]
    if (args.repeats < 1 or not args.clusters or any(k < 1 for k in args.clusters) or
            args.weak_rows_per_worker < 1 or args.weak_iterations < 1):
        parser.error("repeats, cluster counts, rows, and iterations must be positive")

    RAW.mkdir(parents=True, exist_ok=True)
    GRAPHS.mkdir(parents=True, exist_ok=True)
    record_environment()
    cases = plan(args)
    print(f"{len(cases)} runs planned ({args.suite}, {args.dataset}, K={args.clusters})", flush=True)
    for case in cases:
        run_case(case, args.force)
    records = load_saved_records()
    if not report(records):
        print("Correctness check failed; inspect summary.csv and weak_summary.csv", file=sys.stderr)
        return 1
    print(f"Saved {len(records)} runs to {RAW}; graphs to {GRAPHS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
