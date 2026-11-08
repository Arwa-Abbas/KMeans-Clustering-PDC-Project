# Project Plan

The project is intentionally divided into testable phases. A phase should pass
its validation gate before work starts on the next one.

Phases 1-6 were first implemented in C11. The code base was then ported to
C++17, as stated in the proposal, and the gates were rerun on the port: the
synthetic tests pass, and the sequential results match the saved C11 reference
runs in `results/`.

## Phase 1 - Dataset setup and extraction (complete)

Deliverables:

- inspect `HIGGS.csv.gz` without loading it into memory;
- validate 29 numeric columns per row (label + 28 features);
- extract atomically to `dataset/raw/HIGGS.csv`;
- record extraction metadata; and
- document Windows/WSL2 and Ubuntu setup.

Gate passed: the tool compiled with GCC using strict warnings, a 1,000-row
smoke test passed, and a full compressed scan confirmed exactly 11,000,000
valid rows. Materialize the CSV after moving the repository to Ubuntu's Linux
filesystem, immediately before Phase 2 preprocessing.

## Phase 2 - Out-of-core preprocessing (complete)

Deliverables:

- compute per-feature means and standard deviations using a numerically stable
  streaming algorithm;
- exclude the label column;
- write a row-major, z-score-normalized `float32` binary matrix;
- write binary-format metadata (shape, type, byte order, statistics); and
- create a small deterministic subset for fast development.

Gate passed: the full matrix contains exactly `11,000,000 * 28` values and is
1,232,000,000 bytes. The first 1,000 rows were read back and every feature
matched a freshly normalized source value within `1e-6`. A deterministic
one-million-row development matrix was also generated.

## Phase 3 - Sequential reference implementation (complete)

Deliverables:

- binary loader, optionally limited to the first N rows;
- deterministic K-Means++ initialization using a fixed seed and a documented
  100,000-point sample;
- Lloyd iterations using squared Euclidean distance;
- convergence by changed-assignment percentage and iteration cap; and
- SSE, centroid, iteration, and timing output.

Gate passed: a four-point, two-cluster synthetic test reached the hand-checked
SSE of 112. The one-million-row K=16 run converged in 48 iterations, and a
repeat run produced identical centroids, SSE, and iteration count. The full
11-million-row K=16 run converged in 41 iterations with final SSE
216,216,213.267. The default fixed seed is 42 and the first 100,000 points
form the deterministic K-Means++ sample.

## Phase 4 - Shared-memory OpenMP implementation (complete)

Deliverables:

- parallel point-to-centroid assignment through the shared kernel;
- per-thread, cache-line-padded cluster sums and counts to avoid races and
  false sharing;
- controlled thread count and contiguous static row blocks; and
- scaling runs for 1, 2, 4, 8, and 16 threads.

Gate passed: a synthetic comparison with uneven row blocks matches the
sequential centroids and SSE for 1-5 threads, and one thread is bit-for-bit
identical to sequential. On one million HIGGS rows, 1, 2, 4, 8, and 16 threads
all converged in 48 iterations with maximum centroid difference below `3e-16`
and SSE difference below `1e-6` from sequential.

## Phase 5 - Distributed-memory MPI implementation (complete)

Deliverables:

- contiguous row partitioning across ranks, including uneven remainders;
- collective MPI-IO reads from the binary matrix;
- replicated centroids;
- one `MPI_Allreduce` per iteration carrying sums, counts, changed count, and
  SSE packed into one array of doubles; and
- scaling runs for 1, 2, 4, and 8 processes.

Gate passed: a three-rank synthetic test (uneven blocks, full input and a row
prefix) matches sequential. Runs with 1, 2, 3, 4, and 8 processes on one
million HIGGS rows all converged in 48 iterations and matched sequential
centroids within `3e-16`. Four processes on all 11 million rows converged in
41 iterations and matched sequential centroids within `6e-15`.

## Phase 6 - Hybrid MPI + OpenMP implementation (complete)

Deliverables:

- OpenMP assignment within every MPI rank;
- explicit MPI thread support (`MPI_Init_thread`);
- process/thread configurations up to 16 total workers; and
- documented placement: `mpirun --bind-to none`, so Open MPI does not pin a
  two-rank job's ranks to one core each and squeeze their threads onto it.

Gate passed for the tested configurations: 1x4, 2x2, 3x2, and 4x2 on one million
rows all converged in 48 iterations, including the uneven three-rank
partition. A 2x4 full-data run converged in 41 iterations. Final centroids
matched sequential within `5e-15`; final SSE differed by less than `3e-13`
relative. MPI calls occur outside OpenMP worker regions.

## Phase 7 - Experiments and report

Run K = 16, 64, and 256. Strong scaling: the 1-million and 11-million-row
inputs with 1-16 workers. Weak scaling: 1.375 million rows per worker for 1, 2,
4, and 8 workers (up to the full 11 million), with a fixed iteration count.
Exclude loading and one-time preprocessing from the K-Means compute timing.
Record the median of three runs and report:

- runtime;
- speedup `T1 / Tp`;
- efficiency `speedup / (processes * threads)`;
- weak-scaling efficiency `T(1 worker) / T(p workers)`;
- iteration count and final SSE; and
- hardware, compiler, flags, process placement, and thread placement.

Keep the same initialization seed and starting centroids across all variants so
the timing comparison is fair. Results saved by the C11 code are superseded and
must be regenerated with the current runner.
