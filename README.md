# Parallel K-Means Clustering on the HIGGS Dataset

Course project for Parallel and Distributed Computing. The final program will
compare sequential, OpenMP, MPI, and hybrid MPI + OpenMP implementations of
Lloyd's K-Means on the 11-million-row HIGGS dataset.

## Current status

**Phases 1 and 2 are complete:** the archive has been validated and converted
to a normalized binary matrix. K-Means implementation starts in Phase 3.

The repository expects the downloaded archive at:

```text
dataset/HIGGS.csv.gz
```

## Setup after cloning

The dataset and generated binaries are intentionally excluded from Git because
the source archive is about 2.6 GB and the normalized binary is about 1.15 GiB.

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/Arwa-Abbas/KMeans-Clustering-PDC-Project.git
   cd KMeans-Clustering-PDC-Project
   ```

2. Install the Ubuntu dependencies:

   ```bash
   sudo apt update
   sudo apt install -y build-essential openmpi-bin libopenmpi-dev zlib1g-dev
   ```

3. Download `HIGGS.csv.gz` from the
   [UCI HIGGS dataset page](https://archive.ics.uci.edu/dataset/280/higgs),
   create the local dataset directory, and place the file at exactly:

   ```text
   KMeans-Clustering-PDC-Project/dataset/HIGGS.csv.gz
   ```

   From the repository root, the expected relative path is:

   ```text
   dataset/HIGGS.csv.gz
   ```

4. Build and validate the setup:

   ```bash
   make
   make test
   ./build/higgs_dataset inspect --full-scan
   ```

5. Generate the normalized binary used by the K-Means implementations:

   ```bash
   ./build/preprocess_higgs
   ```

   Use `--overwrite` only when intentionally replacing an existing processed
   binary. Keep at least 5 GiB free for the compressed source, processed matrix,
   metadata, and temporary output. CSV extraction is not required.

The checked archive is 2.623 GiB. HIGGS rows have 29 comma-separated numeric
values: a binary class label followed by 28 features. The label is useful for
dataset validation but will not be used as a K-Means feature.

The full validation result was:

```text
rows:      11,000,000
features:  28 (plus one label column)
label 0:   5,170,877
label 1:   5,829,123
status:    valid
```

## Required parallel architecture

The final implementation will be C11 and will retain four executable modes for
a fair comparison:

1. sequential reference;
2. OpenMP shared-memory implementation;
3. MPI distributed-memory implementation; and
4. hybrid MPI + OpenMP implementation.

In the hybrid version, MPI ranks receive contiguous blocks of rows and each
rank uses OpenMP threads for point-to-centroid distance calculations and local
partial sums. `MPI_Allreduce` then combines cluster sums, counts, convergence
information, and SSE across ranks. MPI calls will be made outside OpenMP worker
regions unless a later design explicitly requires a stronger MPI thread level.

## Do I need Ubuntu?

We will use **Ubuntu with C** for the project. On a Windows machine, Ubuntu
24.04 or 22.04 under WSL2 is the simplest environment for GCC, Open MPI,
OpenMP, and MPI-IO. Phase 1 is also written in C so the whole project follows
one language and toolchain.

Open PowerShell as Administrator once:

```powershell
wsl --install -d Ubuntu-24.04
```

After restarting, open Ubuntu and install the future build toolchain:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev zlib1g-dev
```

Keep the repository in the WSL Linux filesystem (for example
`~/projects/kmeans_clustering`) during MPI benchmarks. Running from `/mnt/c`
works, but file I/O is often slower and can distort timing results.

## Phase 1: inspect and extract HIGGS

From an Ubuntu terminal, build the C dataset tool:

```bash
make
```

1. Run a quick structural inspection without decompressing the whole archive:

   ```bash
   ./build/higgs_dataset inspect
   ```

2. Run the Phase 1 smoke test (build plus a 1,000-row inspection):

   ```bash
   make test
   ```

3. CSV extraction is optional because the Phase 2 preprocessor reads the gzip
   archive directly. To materialize the CSV, use:

   ```bash
   ./build/higgs_dataset extract
   ```

   The command writes to `dataset/raw/HIGGS.csv.part` first and renames it to
   `dataset/raw/HIGGS.csv` only after successful validation. It also writes
   `dataset/raw/HIGGS.metadata.json`. If a previous partial extraction exists,
   rerun with `--overwrite`.

4. Optionally perform a full validation without keeping an extracted CSV:

   ```bash
   ./build/higgs_dataset inspect --full-scan
   ```

Useful options (place them after `inspect` or `extract`):

```text
--archive PATH       use a different .csv.gz archive
--output PATH        choose the extracted CSV path
--expected-rows N    change the expected row count
--overwrite          replace an existing output or partial file
```

The tool reads the gzip archive through zlib without loading it into memory.
The full archive has already passed a no-output validation scan. The extracted
CSV is not retained because Phase 2 does not require it. Do not commit generated
dataset files.

## Phase 2: normalized binary dataset

Build and run the two-pass C preprocessor:

```bash
make
./build/preprocess_higgs --overwrite
```

It streams the compressed source twice. The first pass computes per-feature
means and population standard deviations using Welford's algorithm. The second
pass excludes the label, z-score normalizes the 28 features, converts them to
`float32`, and writes a headerless row-major matrix atomically.

Generated outputs:

```text
dataset/processed/higgs_f32.bin           11,000,000 rows (1,232,000,000 bytes)
dataset/processed/higgs_f32.metadata.json shape, layout, means, and deviations
dataset/processed/higgs_1m_f32.bin        deterministic development subset
dataset/processed/higgs_1m_f32.metadata.json
```

The preprocessor verifies the exact binary size and reads back the first 1,000
rows to compare every stored feature against a freshly normalized source value.
See [docs/BINARY_FORMAT.md](docs/BINARY_FORMAT.md) for the MPI-IO offset rule.

## Planned phases

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for phase deliverables,
validation gates, and the point at which MPI and OpenMP are introduced.

## Source layout

```text
include/kmeans/              public C headers shared by modules
src/dataset/                 HIGGS validation, extraction, and streaming reader
src/preprocessing/           statistics, normalization orchestration, binary I/O
src/tools/                   command-line entry points
tests/                       focused C unit tests
dataset/                     source and generated data (generated data ignored)
docs/                        design and phase documentation
build/                       generated objects and executables (ignored)
```

Modules are separated by responsibility; small functions that belong to one
responsibility remain together instead of creating a file for every function.

## Dataset citation

HIGGS is available from the UCI Machine Learning Repository. The project uses
the 28 feature columns and excludes the first label column from clustering.

