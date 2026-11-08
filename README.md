# Parallel K-Means Clustering on Large Dataset (HIGGS Dataset)

A Parallel and Distributed Computing (PDC) course project that clusters the
11-million-point HIGGS dataset with K-Means in C. The main implementation is
hybrid: MPI processes divide the data, and OpenMP threads calculate distances
within each process. Sequential, OpenMP-only, and MPI-only programs provide
correctness and performance comparisons.

## Features

- Four K-Means execution modes: sequential, OpenMP, MPI, and MPI + OpenMP.
- C11 implementation with GCC, Open MPI, OpenMP, zlib, and GNU Make.
- Streaming validation and z-score normalization of the HIGGS archive.
- Compact row-major `float32` matrix with a metadata file for MPI-IO.
- Fixed-seed K-Means++ initialization from the first 100,000 points.
- Squared Euclidean distance, configurable K and iteration limit, and a
  changed-assignment stopping rule (0.1% by default).
- Per-run JSON with initial/final centroids, SSE, iterations, and separate
  loading, initialization, and computation timings.
- Synthetic correctness tests and saved HIGGS reference results.

The full analysis of K=16, 64, and 256, repeated benchmark runs, speedup,
efficiency, and plots is the remaining experimental work. See
[the project plan](docs/PROJECT_PLAN.md) for its scope.

## How the hybrid version works

1. MPI-IO loads a contiguous, non-overlapping block of points on each rank.
   Remainder rows are assigned to the first ranks.
2. Every rank starts with the same centroids broadcast from rank 0.
3. OpenMP threads on each rank assign points to their nearest centroids and
   accumulate private cluster sums, counts, SSE, and changed assignments.
4. Each rank combines its threads' values. `MPI_Allreduce` then combines the
   values across ranks so all ranks update to the same centroids.
5. The loop ends when fewer than 0.1% of assignments change or the iteration
   limit is reached. MPI calls occur outside OpenMP regions.

The sequential, OpenMP, MPI, and hybrid modes use the same dataset, seed,
starting centroids, and stopping rule for comparable results.

## Requirements

Use Ubuntu 24.04 or 22.04. On Windows, Ubuntu under WSL2 works. Install it
from an Administrator PowerShell window only if it is not already installed:

```powershell
wsl --install -d Ubuntu-24.04
```

Enter Ubuntu with `wsl -d Ubuntu-24.04`, then install the build tools:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev zlib1g-dev
```

For meaningful performance measurements, place the repository in Ubuntu's
Linux filesystem (for example `~/projects/KMeans-Clustering-PDC-Project`).
Running from `/mnt/c` works, but its file I/O can distort timing.

## Setup after cloning

Run these commands in the Ubuntu terminal:

```bash
git clone https://github.com/Arwa-Abbas/KMeans-Clustering-PDC-Project.git
cd KMeans-Clustering-PDC-Project
mkdir -p dataset results
```

Download `HIGGS.csv.gz` from the
[UCI HIGGS dataset page](https://archive.ics.uci.edu/dataset/280/higgs) and
place it at `dataset/HIGGS.csv.gz`. It has 11 million rows: the first CSV
column is a label, followed by 28 features. The label is excluded from
clustering. Dataset files are excluded from Git because of their size.

Build and test:

```bash
make
make test
./build/higgs_dataset inspect --full-scan
```

`make test` should end with `online statistics test passed`,
`sequential test passed`, and `OpenMP comparison passed`. The full scan should
report 11,000,000 valid rows and 28 features.

Create the binary input once (allow several gigabytes of free disk space):

```bash
./build/preprocess_higgs
```

This writes `dataset/processed/higgs_f32.bin` and its metadata. The full binary
has exactly `11,000,000 * 28 * 4 = 1,232,000,000` bytes. CSV extraction is
optional; the preprocessor reads the compressed archive directly. Use
`--overwrite` only when intentionally regenerating an existing binary. For
the binary layout and MPI-IO offset, see [the format description](docs/BINARY_FORMAT.md).

Create the one-million-row development input used by the commands below:

```bash
./build/preprocess_higgs --max-rows 1000000 \
  --output dataset/processed/higgs_1m_f32.bin \
  --metadata dataset/processed/higgs_1m_f32.metadata.json
```

This subset is normalized using statistics from its own first million rows.

## Run K-Means

These examples use the one-million-row development binary, K=16, seed 42,
and a maximum of 100 iterations. They can be run independently:

```bash
./build/kmeans_sequential --clusters 16 \
  --output results/sequential_1m_k16.json

./build/kmeans_openmp --threads 4 --clusters 16 \
  --output results/openmp_1m_k16_t4.json

mpirun -np 4 ./build/kmeans_mpi --clusters 16 \
  --output results/mpi_1m_k16_p4.json

mpirun -np 2 ./build/kmeans_hybrid --threads 2 --clusters 16 \
  --output results/hybrid_1m_k16_p2_t2.json
```

In the last command, `-np 2` launches two MPI processes and `--threads 2`
starts two OpenMP threads inside each process: four workers in total.
`--clusters 16` is the number of groups K-Means forms; it is unrelated to
thread count.

Each run prints a line per iteration with the changed assignments, changed
fraction, assignment SSE, and empty-cluster count. The final line reports rows,
K, iterations, final SSE, and timings. The default stopping threshold is
`--tolerance 0.001`, meaning 0.1% of points changed clusters. The final SSE is
recalculated against the final centroids.

To use all 11 million points, add these options to any of the commands above:

```bash
--data dataset/processed/higgs_f32.bin \
--metadata dataset/processed/higgs_f32.metadata.json
```

Other options are `--clusters`, `--max-iterations`, `--tolerance`, `--seed`,
and `--init-sample-size`. The OpenMP and hybrid programs also accept
`--threads`. Results from runs with the same input and settings should match
in SSE and centroids within floating-point tolerance. Computation times vary
with hardware and process/thread placement; use repeated runs for benchmarks.

## Experiments and graphs

The [experiment guide](experiments/README.md) explains how to install the
Python plotting dependency from `requirements.txt` and run resumable
three-repeat benchmarks. The runner saves raw JSON, a median-time/speedup/
efficiency CSV, and Matplotlib PNG graphs under `experiments/`.

Measured one-million-row results are currently available for the K=16 scaling
grid and a K=64 comparison of the four execution modes. K=64 reached the
100-iteration cap in every measured run; K=256 and full-data benchmark grids
have not been measured yet. The runner can resume those longer experiments
without repeating completed runs. No final report document is included.

## Project structure

```text
include/kmeans/      Public C headers
src/dataset/         HIGGS archive reading and validation
src/preprocessing/   Feature statistics and binary conversion
src/kmeans/          Sequential, OpenMP, MPI, and hybrid computation
src/tools/           Command-line entry points
tests/               Correctness tests
docs/                Project plan and binary format
experiments/         Benchmark runner, measurements, and PNG graphs
results/             Saved run results (tracked in Git)
dataset/             Local source and processed data (ignored by Git)
build/               Generated executables and objects (ignored by Git)
```

## Dataset citation

Whiteson, D. (2014). *HIGGS*. UCI Machine Learning Repository.
[https://doi.org/10.24432/C5V312](https://doi.org/10.24432/C5V312)

PDC Course Project Made by Arwa Abbas and Nabira Khan.
