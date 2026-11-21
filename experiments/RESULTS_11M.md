# Eleven-Million-Row Strong-Scaling Results

The version-2 strong-scaling experiment is complete for the full
eleven-million-row HIGGS dataset at `K=16` and `K=64`. Each configuration was
measured three times. Reported times are medians of the K-Means computation
only; dataset loading and centroid initialization are excluded.

All 28 summarized configurations matched their sequential reference within
the configured SSE and centroid tolerances. The K=16 runs converged in 41
iterations. The K=64 runs reached the configured 100-iteration cap and
all four implementations produced equivalent final SSE and centroids within
the configured correctness tolerances.

## Summary

| K | Sequential median | Fastest configuration | Fastest median | Speedup | Efficiency |
|---:|---:|:---|---:|---:|---:|
| 16 | 50.446 s | OpenMP, 16 threads | 5.272 s | 9.569x | 59.8% |
| 64 | 444.020 s | OpenMP, 16 threads | 48.951 s | 9.071x | 56.7% |

## K=64 implementation comparison

| Implementation | Best configuration | Median | Speedup | Efficiency |
|:---|:---|---:|---:|---:|
| OpenMP | 16 threads | 48.951 s | 9.071x | 56.7% |
| Hybrid | 4 processes x 4 threads | 67.268 s | 6.601x | 41.3% |
| MPI | 8 processes | 117.123 s | 3.791x | 47.4% |

OpenMP remained the fastest implementation. The K=64 sequential computation
took 444.020 seconds, while OpenMP with 16 threads reduced it to 48.951
seconds. Hybrid was the second-fastest implementation, and MPI had the lowest
speedup on this single shared-memory system.

The complete rows, including processes, threads, total workers, median time,
speedup, efficiency, SSE error, centroid error, and correctness, are in
[`summary.csv`](summary.csv). The timing, speedup, efficiency, and scaling
plots are in [`graphs/`](graphs/) as `11m_k16_*.png` and `11m_k64_*.png`.

Per-run JSON records are intentionally kept local under `experiments/raw/`.
They allow interrupted experiments to resume, but are not committed to Git.
