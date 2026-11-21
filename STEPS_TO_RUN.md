# Steps to Run

A checklist from a fresh Windows PC to results and graphs. After step 1, run
every command in the Ubuntu terminal from the repository root.

## 1. Install Ubuntu (Windows only)

In an Administrator PowerShell window:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if asked, open Ubuntu from the Start menu, and create a user.

## 2. Install the tools

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev zlib1g-dev python3-venv unzip
```

## 3. Get the code

```bash
cd ~
git clone https://github.com/Arwa-Abbas/KMeans-Clustering-PDC-Project.git
cd KMeans-Clustering-PDC-Project
```

Keep the repository in the Linux filesystem (`~`), not under `/mnt/c`, so file
access does not distort timings.

## 4. Build and test

```bash
make
make test
```

`make test` needs no dataset and should end with `online statistics test
passed`, `sequential test passed`, `OpenMP comparison passed`, and `MPI
comparison passed`.

## 5. Download HIGGS (about 2.8 GB)

```bash
mkdir -p dataset
wget -O dataset/higgs.zip https://archive.ics.uci.edu/static/public/280/higgs.zip
unzip dataset/higgs.zip -d dataset      # creates dataset/HIGGS.csv.gz
rm dataset/higgs.zip
./build/higgs_dataset inspect --full-scan
```

The scan should report 11,000,000 valid rows. The UCI server does not resume
interrupted downloads: if the download stops partway, delete the file and
start again.

## 6. Create the binary matrices

```bash
./build/preprocess_higgs
./build/preprocess_higgs --max-rows 1000000 \
  --output dataset/processed/higgs_1m_f32.bin \
  --metadata dataset/processed/higgs_1m_f32.metadata.json
```

The first command writes the full 11M-row matrix (1.23 GB) and takes several
minutes; the second writes the 1M-row development matrix.

## 7. Run each version once (demo)

```bash
./build/kmeans_sequential --clusters 16
./build/kmeans_openmp --threads 4 --clusters 16
mpirun --bind-to none -np 4 ./build/kmeans_mpi --clusters 16
mpirun --bind-to none -np 2 ./build/kmeans_hybrid --threads 2 --clusters 16
```

All four use the 1M-row matrix and should finish with `iterations=48` and
`final_sse=19656447.8453`; only the timings differ.

## 8. Set up Python for the experiments

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 9. Run the experiments

Before long runs, keep the test system on stable power, use a consistent power
mode, and close other programs.

| Command | Measures | Observed or estimated time |
|---|---|---:|
| `.venv/bin/python experiments/run.py --suite quick --clusters 16` | Four configurations, as a check | 1-2 min |
| `.venv/bin/python experiments/run.py --suite scaling --dataset 1m --clusters 16,64,256` | Strong scaling, 1M rows | about 1 h 45 min |
| `.venv/bin/python experiments/run.py --suite weak --clusters 16,64` | Weak scaling, up to 11M rows | about 22 min |
| `.venv/bin/python experiments/run.py --suite scaling --dataset 11m --clusters 16` | Strong scaling, 11M rows, K=16 (complete) | about 17 min measured compute time |
| `.venv/bin/python experiments/run.py --suite scaling --dataset 11m --clusters 64` | Strong scaling, 11M rows, K=64 (complete) | about 170 min measured compute time |
| `.venv/bin/python experiments/run.py --suite scaling --dataset 11m --clusters 256` | Strong scaling, 11M rows, K=256 | about 14 h |

Every suite repeats each configuration three times. Runs are resumable:
rerunning a command skips finished repetitions. Add `--force` to rerun them.

## 10. Find the outputs

```text
experiments/summary.csv          Strong scaling: median time, speedup, efficiency, correctness
experiments/weak_summary.csv     Weak scaling: median time, weak efficiency, correctness
experiments/graphs/              Bar charts plus *_scaling.png and weak_*_efficiency.png
experiments/environment.json     Platform, tools, MPI placement, worker limit
experiments/raw/                 One JSON file per run (local only, not committed)
```

To redraw the graphs from the two CSV files without running anything:

```bash
.venv/bin/python experiments/run.py --redraw
```

## Troubleshooting

- **`make` fails with `missing separator` or `\r` errors:** the files have
  Windows line endings. Clone the repository inside Ubuntu (step 3).
- **`mpirun` reports not enough slots:** the machine has fewer cores than
  `-np`. Add `--oversubscribe` for testing only, not for timings.
- **`mpirun` refuses to run as root** (for example in Docker): run
  `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` first.
- **`does not match the size in ...metadata.json`:** preprocessing was
  interrupted. Rerun step 6 with `--overwrite`.
- **`--rows ... exceeds the ... rows`:** the weak suite needs the full 11M-row
  matrix from step 6.
- **One-worker timings vary, or efficiency is above 1:** CPU frequency, cache
  state, background activity, and operating-system scheduling affect short
  measurements. See "Fair comparisons" in
  [experiments/README.md](experiments/README.md).
