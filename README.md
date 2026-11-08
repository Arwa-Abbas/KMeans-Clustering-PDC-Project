# Parallel K-Means Clustering on the HIGGS Dataset

A Parallel and Distributed Computing project that implements Lloyd's K-Means
in C and compares four execution models on a large, high-dimensional dataset:

1. sequential C;
2. OpenMP shared-memory parallelism;
3. MPI distributed-memory parallelism; and
4. hybrid MPI + OpenMP parallelism.

The project uses the 11-million-row HIGGS dataset to study correctness,
runtime, speedup, efficiency, scalability, and different cluster counts.

## Project objectives

- Process a dataset that is too large for convenient CSV-based experiments.
- Keep all implementations algorithmically comparable and deterministic.
- Distribute contiguous point blocks across MPI processes.
- Parallelize point-to-centroid distance calculations with OpenMP.
- Combine cluster sums, counts, SSE, and convergence data with MPI collectives.
- Compare sequential, OpenMP, MPI, and hybrid performance.
- Report runtime, speedup, efficiency, convergence, and final SSE.

## Project phases

| Phase | Work | Status |
|---|---|---|
| 1 | Dataset inspection, validation, and safe extraction tooling | **Done** |
| 2 | Streaming normalization and `float32` binary conversion | **Done** |
| 3 | Deterministic sequential K-Means reference implementation | **Done** |
| 4 | Shared-memory K-Means using OpenMP | **Done** |
| 5 | Distributed-memory K-Means using MPI and MPI-IO | **Done** |
| 6 | Hybrid MPI + OpenMP K-Means | **Done** |
| 7 | Experiments, scalability analysis, graphs, and final report | **Remaining** |

Detailed deliverables and validation gates are documented in
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## Parallel design

In the final hybrid implementation, each MPI process will load a contiguous
block of normalized points through MPI-IO. Every process will keep a replicated
copy of the centroids. OpenMP threads inside each process will perform local
point assignment and maintain thread-local cluster sums and counts.

After each iteration, `MPI_Allreduce` will combine the local sums, counts, SSE,
and number of changed assignments. All four implementations will use the same
initial centroids, convergence rule, empty-cluster policy, and input data.

## Dataset

The project uses the
[HIGGS dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs).

```text
Rows:              11,000,000
CSV columns:       29
Label columns:      1 (excluded from clustering)
Feature columns:   28
Compressed size:   approximately 2.6 GB
Processed size:    1,232,000,000 bytes
Processed type:    little-endian float32
Processed layout:  headerless row-major matrix
```

The full local archive validation produced:

```text
label 0:  5,170,877 rows
label 1:  5,829,123 rows
status:   valid
```

Dataset files and generated binaries are intentionally excluded from Git.

## Platform and requirements

The project targets Ubuntu with GCC, OpenMP, Open MPI, MPI-IO, GNU Make, and
zlib. Ubuntu 24.04 or 22.04 under WSL2 is recommended for Windows users.

Install WSL from an Administrator PowerShell window if necessary:

```powershell
wsl --install -d Ubuntu-24.04
```

Install the project dependencies inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev zlib1g-dev
```

For meaningful I/O benchmarks, keep the repository in the Ubuntu filesystem,
such as `~/projects/KMeans-Clustering-PDC-Project`. Access through `/mnt/c`
works but can be considerably slower.

## Setup after cloning

1. Clone the repository:

   ```bash
   git clone https://github.com/Arwa-Abbas/KMeans-Clustering-PDC-Project.git
   cd KMeans-Clustering-PDC-Project
   ```

2. Create the dataset directory:

   ```bash
   mkdir -p dataset
   ```

3. Download `HIGGS.csv.gz` from the
   [official UCI page](https://archive.ics.uci.edu/dataset/280/higgs) and place
   it at this exact relative path:

   ```text
   dataset/HIGGS.csv.gz
   ```

4. Build and run the tests:

   ```bash
   make
   make test
   ```

5. Validate the complete compressed dataset:

   ```bash
   ./build/higgs_dataset inspect --full-scan
   ```

6. Generate the normalized binary matrix:

   ```bash
   ./build/preprocess_higgs
   ```

   Use `--overwrite` only when intentionally replacing an existing processed
   binary. Keep at least 5 GiB free. Full CSV extraction is not required.

## Available dataset tools

Inspect a 1,000-row sample:

```bash
./build/higgs_dataset inspect
```

Validate all 11 million compressed rows without keeping a CSV:

```bash
./build/higgs_dataset inspect --full-scan
```

Optionally extract the CSV atomically:

```bash
./build/higgs_dataset extract
```

Generate or replace the processed matrix:

```bash
./build/preprocess_higgs --overwrite
```

The preprocessor performs two streaming passes. The first uses Welford's
algorithm to calculate the mean and population standard deviation of every
feature. The second excludes the label, applies z-score normalization, and
writes each value as `float32`.

Generated outputs:

```text
dataset/processed/higgs_f32.bin
dataset/processed/higgs_f32.metadata.json
dataset/processed/higgs_1m_f32.bin
dataset/processed/higgs_1m_f32.metadata.json
```

The complete binary must contain exactly:

```text
11,000,000 * 28 * 4 = 1,232,000,000 bytes
```

The preprocessor checks the exact size and compares the first 1,000 stored rows
with freshly parsed and normalized source values. The binary layout and future
MPI-IO offset calculation are documented in
[docs/BINARY_FORMAT.md](docs/BINARY_FORMAT.md).

## Sequential K-Means reference

The default run uses the one-million-row development matrix, K=16, seed 42,
100,000 initialization samples, a 0.1% changed-assignment threshold, and a
100-iteration cap:

```bash
./build/kmeans_sequential --output results/sequential_1m_k16.json
```

Create `results/` first with `mkdir -p results`. To use the complete dataset:

```bash
./build/kmeans_sequential \
  --data dataset/processed/higgs_f32.bin \
  --metadata dataset/processed/higgs_f32.metadata.json \
  --clusters 16 \
  --output results/sequential_11m_k16.json
```

Options include `--clusters`, `--seed`, `--init-sample-size`,
`--max-iterations`, and `--tolerance` (a fraction, so `0.001` is 0.1%). The
result JSON contains starting and final centroids, final SSE, iteration count,
and separate loading, initialization, and iteration timings. The printed SSE
for an iteration uses centroids before that iteration's update; `final_sse` is
recalculated against the final centroids. Loading and initialization are
excluded from the measured computation time.

The sequential reference passed the synthetic test, converged in 48 iterations
on one million points, and converged in 41 iterations on the full 11 million
points at K=16. A repeat run on one million points produced identical final
centroids and SSE. Additional K values are part of Phase 7 experiments.

## OpenMP K-Means

The OpenMP executable shares the matrix loader, fixed-seed K-Means++
initialization, and result format with the sequential reference. Each thread
assigns a static block of points and accumulates private cluster sums, counts,
changed assignments, and SSE. These are combined after the parallel region.

```bash
mkdir -p results
./build/kmeans_openmp --threads 4 --clusters 16 \
  --output results/openmp_1m_k16_t4.json
```

For the full dataset, supply `--data dataset/processed/higgs_f32.bin` and
`--metadata dataset/processed/higgs_f32.metadata.json`. Results for 1, 2, 4,
8, and 16 threads on one million rows, plus a four-thread full-data result, are
stored in `results/`. All matched the sequential iteration counts; maximum
centroid differences were below `6e-15`. The recorded timings are individual
runs on WSL and are not the median-of-three benchmark results planned for
Phase 7.

## MPI K-Means

The MPI executable uses collective MPI-IO to read contiguous, non-overlapping
row blocks. The first ranks receive one extra row when the dataset size is not
divisible by the process count. Rank 0 initializes the shared centroids from
the same first 100,000 points and fixed seed as the other variants, then
broadcasts them. Each rank assigns its local points; `MPI_Allreduce` combines
cluster sums, counts, changed assignments, and SSE after each iteration.

```bash
mpirun -np 4 ./build/kmeans_mpi --clusters 16 \
  --output results/mpi_1m_k16_p4.json
```

For all 11 million rows:

```bash
mpirun -np 4 ./build/kmeans_mpi \
  --data dataset/processed/higgs_f32.bin \
  --metadata dataset/processed/higgs_f32.metadata.json \
  --clusters 16 --output results/mpi_11m_k16_p4.json
```

The one-million-row runs with 1, 2, 3, 4, and 8 ranks converged in 48
iterations; the three-rank case also checks uneven row partitioning. The
four-rank full-data run converged in 41 iterations. Their final centroids
matched the sequential reference within `6e-15`. Timings are individual WSL
runs; formal repeat-run benchmarks belong to Phase 7.

## Hybrid MPI + OpenMP K-Means

The hybrid executable combines MPI row partitioning and MPI-IO with OpenMP
point assignment inside every rank. Each OpenMP thread uses private cluster
sums, counts, changed-assignment counts, and SSE; a rank combines these before
the MPI reductions. MPI calls are made by the main thread outside the OpenMP
regions (`MPI_THREAD_FUNNELED`).

```bash
mpirun -np 2 ./build/kmeans_hybrid --threads 2 --clusters 16 \
  --output results/hybrid_1m_k16_p2_t2.json
```

This example runs two MPI processes with two OpenMP threads in each process:
four workers in total. For the full dataset, pass the same `--data` and
`--metadata` paths shown in the MPI example. The checked 2x2, 3x2, and 4x2
one-million-row runs matched the sequential 48-iteration result. A 2x4 run on
all 11 million rows matched the sequential 41-iteration result with maximum
centroid difference below `5e-15`. Runtime is sensitive to CPU placement and
WSL file I/O, so use the controlled repeated experiments in Phase 7 to draw
speedup conclusions.

## Project structure

```text
KMeans-Clustering-PDC-Project/
|-- include/kmeans/       Public C headers
|-- src/dataset/          HIGGS validation and streaming reader
|-- src/preprocessing/    Statistics, normalization, and binary output
|-- src/kmeans/            Matrix loader, Lloyd loops, MPI, local OpenMP assignment
|-- src/tools/            Command-line program entry points
|-- tests/                Focused C unit tests
|-- docs/                 Project plan and binary format documentation
|-- dataset/              Local source/processed data (ignored by Git)
|-- results/              Reference runs and experiment output (tracked in Git)
|-- build/                Compiled objects and executables (ignored by Git)
|-- Makefile
`-- README.md
```

Source files are separated by functionality. Closely related functions remain
together rather than creating a separate file for every small function.

## Reproducibility and performance rules

- Use a fixed seed and identical starting centroids for every implementation.
- Exclude data loading and preprocessing from K-Means compute timing.
- Use squared Euclidean distance consistently.
- Check parallel centroids and SSE against the sequential reference within a
  documented floating-point tolerance.
- Report the median of three benchmark runs.
- Record compiler version, flags, hardware, process count, thread count, and
  process/thread placement.
- Treat recorded timing values as machine-specific; centroids and SSE are the
  deterministic correctness reference for runs with matching data and settings.
- Calculate speedup as `T1 / Tp` and efficiency as
  `speedup / (MPI processes * OpenMP threads)`.

## Dataset citation

Whiteson, D. (2014). *HIGGS*. UCI Machine Learning Repository.
[https://doi.org/10.24432/C5V312](https://doi.org/10.24432/C5V312)
