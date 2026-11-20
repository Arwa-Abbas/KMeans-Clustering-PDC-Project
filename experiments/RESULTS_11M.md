# Eleven-Million-Row Strong-Scaling Results

The version-2 strong-scaling experiment is complete for the full
eleven-million-row HIGGS dataset at `K=16`. Each configuration was measured
three times. Reported times are medians of the K-Means computation only;
dataset loading and centroid initialization are excluded.

All 14 summarized K=16 configurations converged in 41 iterations and matched
the sequential reference within the configured SSE and centroid tolerances.

## K=16 summary

| Measurement | Result |
|:---|:---|
| Sequential median | 50.446 s |
| Fastest configuration | OpenMP, 16 threads |
| Fastest median | 5.272 s |
| Fastest speedup | 9.569x |
| Fastest efficiency | 59.8% |
| Fastest MPI configuration | MPI, 8 processes: 11.900 s, 4.239x |
| Fastest hybrid configuration | Hybrid, 4 processes x 4 threads: 7.078 s, 7.127x |

The complete rows, including processes, threads, total workers, median time,
speedup, efficiency, SSE error, centroid error, and correctness, are in
[`summary.csv`](summary.csv). The timing, speedup, efficiency, and scaling
plots are in [`graphs/`](graphs/) as `11m_k16_*.png`.

Per-run JSON records are intentionally kept local under `experiments/raw/`.
They allow interrupted experiments to resume, but are not committed to Git.
