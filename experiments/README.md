# Experiments

The C++ executables perform clustering. `run.py` runs them repeatedly, checks
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
processes, and hybrid with two processes and two threads each.

## Strong scaling

The dataset stays fixed while workers increase:

```bash
.venv/bin/python experiments/run.py --suite scaling --dataset 1m --clusters 16,64,256 --repeats 3
.venv/bin/python experiments/run.py --suite scaling --dataset 11m --clusters 16,64,256 --repeats 3
```

The scaling suite runs sequential, OpenMP 1/2/4/8/16 threads, MPI 1/2/4/8
processes, and hybrid 2x2, 2x4, 4x2, 4x4. Speedup is the median sequential
computation time divided by the median configuration time. Efficiency divides
speedup by MPI processes times OpenMP threads.

## Weak scaling

Each worker keeps the same number of rows, so ideal weak scaling keeps the
time constant as workers increase:

```bash
.venv/bin/python experiments/run.py --suite weak --clusters 16,64 --repeats 3
```

By default every worker gets 1,375,000 rows of the full 11M binary, so 1, 2,
4, and 8 workers cluster 1.375M, 2.75M, 5.5M, and 11M rows. The rows are
prefixes of the full binary (`--rows`), so all sizes share one normalization.
Larger inputs need more iterations to converge, so weak-scaling runs use a fixed
iteration count (`--weak-iterations`, default 20, with tolerance 0) to compare
the same amount of work per row. Weak-scaling efficiency is the one-worker
sequential time on 1.375M rows divided by the configuration's time. The
sequential program also runs at every size, and each configuration is checked
against the sequential result for the same rows.

## Run times

Each sequential run on all 11M rows takes roughly 2-3 minutes at K=16, 20
minutes at K=64, and 1.5 hours at K=256 on an 8-core laptop, so the full K=256
grid is an overnight job. Each successful repetition is saved immediately
under `experiments/raw/`. Rerunning the same command skips those repetitions
and continues unfinished work. Use `--force` to rerun them deliberately.

## Outputs

```text
experiments/raw/                 Per-run result JSON and command metadata
experiments/summary.csv          Strong scaling: median time, speedup, efficiency, correctness
experiments/weak_summary.csv     Weak scaling: median time, time per iteration, efficiency
experiments/graphs/*.png         Runtime, speedup, efficiency, and scaling graphs
experiments/environment.json     CPU, cores, tool versions, MPI placement, data sizes
```

Loading and centroid initialization are excluded from `computation_seconds`.
Correctness compares final SSE and centroids against the sequential run for
the same rows, K, and seed.

## Fair comparisons

- All four programs call the same compiled assignment kernel, so the sequential
  program, OpenMP with one thread, and MPI with one process do the same work
  with the same machine code.
- MPI programs are launched with `mpirun --bind-to none`. Without it, Open MPI
  pins each rank to one core when it starts two or fewer ranks, and a hybrid
  rank's OpenMP threads would then share that core.
- Records carry a results version. Runs saved by older code (before the C++17
  port and these fixes) are ignored and rerun automatically.

Timings on `/mnt/c` may reflect Windows-mounted filesystem behavior and should
be labeled as WSL results. The C++ toolchain requires the Ubuntu packages
`build-essential`, `openmpi-bin`, `libopenmpi-dev`, and `zlib1g-dev`.
`requirements.txt` covers Python plotting dependencies only.
