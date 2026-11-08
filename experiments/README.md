# Experiments

The C executables perform clustering. `run.py` runs them repeatedly, checks
parallel results against the sequential reference, and writes CSV summaries
and PNG graphs using Matplotlib.

From the repository root in Ubuntu:

```bash
make
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/run.py --suite quick --dataset 1m --clusters 16 --repeats 3
```

The quick suite includes sequential, OpenMP with four threads, MPI with two
processes, and hybrid with two processes and two threads each. For a broader
scaling experiment:

```bash
.venv/bin/python experiments/run.py --suite scaling --dataset 1m --clusters 16,64,256 --repeats 3
```

For the complete 11-million-row dataset:

```bash
.venv/bin/python experiments/run.py --suite scaling --dataset 11m --clusters 16,64,256 --repeats 3
```

The scaling suite runs sequential, OpenMP 1/2/4/8/16 threads, MPI 1/2/4/8
processes, and hybrid 2x2, 2x4, 4x2, 4x4. Full K=256 runs can take a long
time. Each successful repetition is saved immediately under `experiments/raw/`.
Rerunning the same command skips those repetitions and continues unfinished
work. Use `--force` to rerun them deliberately.

Outputs:

```text
experiments/raw/                 Per-run result JSON and command metadata
experiments/summary.csv          Median time, speedup, efficiency, correctness
experiments/graphs/*.png         Runtime, speedup, and efficiency graphs
experiments/environment.json     Hardware, tool versions, and data sizes
```

Loading and centroid initialization are excluded from `computation_seconds`.
Speedup is the median sequential computation time divided by the median
configuration time. Efficiency divides speedup by MPI processes times OpenMP
threads. Correctness compares final SSE and centroids against the sequential
run for the same dataset size, K, and seed. Timings on `/mnt/c` may reflect
Windows-mounted filesystem behavior and should be labeled as WSL results.
The C toolchain requires Ubuntu packages: `build-essential`, `openmpi-bin`,
`libopenmpi-dev`, and `zlib1g-dev`. `requirements.txt` covers Python plotting
dependencies only.

Current measured data includes three repetitions of the complete one-million
row K=16 scaling grid (42 runs) and the one-million-row K=64 quick grid (12
runs). The K=64 runs reached the 100-iteration limit, so compare their results
as capped runs. K=256 and 11-million-row experiment grids remain to be run.
