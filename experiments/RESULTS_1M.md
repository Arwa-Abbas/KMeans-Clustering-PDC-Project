# One-Million-Row Strong-Scaling Results

The version-2 strong-scaling experiment is complete for the one-million-row
HIGGS dataset at `K=16`, `K=64`, and `K=256`. Each configuration was measured
three times. Reported times are medians of the K-Means computation only;
dataset loading and centroid initialization are excluded.

All 42 summarized configurations matched their sequential reference within
the configured SSE and centroid tolerances.

## Summary

| K | Sequential median (s) | Fastest configuration | Fastest median (s) | Speedup |
|---:|---:|:---|---:|---:|
| 16 | 11.236 | OpenMP, 16 threads | 1.047 | 10.737x |
| 64 | 87.821 | OpenMP, 16 threads | 6.582 | 13.343x |
| 256 | 327.691 | OpenMP, 16 threads | 18.539 | 17.675x |

The complete 42-row table, including processes, threads, total workers,
median time, speedup, efficiency, SSE error, centroid error, and correctness,
is in [`summary.csv`](summary.csv). The twelve corresponding timing, speedup,
efficiency, and scaling plots are in [`graphs/`](graphs/).

Efficiencies slightly above 1.0 are retained as measured. They can occur from
cache effects, CPU frequency changes, and normal timing variation; they are
not clamped during analysis.

## Environment

- CPU: Intel Core Ultra 9 275HX, 24 logical CPUs
- Platform: Ubuntu 24.04 under WSL2
- Compiler: GCC 13.3.0 with C++17
- MPI: Open MPI 4.1.6 with `--bind-to none`
- Code revision measured: `4ffc360a8aac71077cf2b29bd46b17c231366b29`

Per-run JSON records are intentionally kept local under `experiments/raw/`.
They can be regenerated with the commands documented in this directory's
README.
