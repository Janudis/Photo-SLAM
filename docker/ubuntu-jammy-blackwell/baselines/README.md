# Blackwell baseline environments

These images isolate the incompatible native dependencies of MonoGS,
HI-SLAM2, and TANDEM from Photo-SLAM. Each image clones a fixed upstream
revision and targets CUDA 12.8 / `sm_120` for the RTX 5080.

The compatibility changes are limited to newer compiler, CUDA, PyTorch, and
headless-output support. Dataset and result paths are mounted at `/datasets`
and `/results`; input data are always read-only.

Do not run these methods concurrently on the 16 GB RTX 5080. Build and create
one method at a time from the host shell:

```bash
cd docker/ubuntu-jammy-blackwell/baselines
export PHOTOSLAM_DATA_ROOT="/media/v4rl/T7 Shield/Dimitris/Datasets_indoor"
export PHOTOSLAM_RESULTS_ROOT="/media/v4rl/T7 Shield/Dimitris/server_benchmark_results"

./build_image.sh monogs
./create_container.sh monogs
```

Run an office0 smoke test without opening an interactive shell:

```bash
./run_container.sh monogs \
  python3 /opt/photoslam-benchmark/run.py monogs replica \
    --sequences office0 \
    --run-id replica-monogs-office0-smoke
```

The launcher prints an immediate job header followed by elapsed time and GPU
usage every 30 seconds. Full native output is written to the trial's
`console.log`. An interactive `Ctrl+C` terminates the native process group;
using an already-active run ID is rejected.

After the smoke test succeeds, run all eight Replica scenes:

```bash
./run_container.sh monogs \
  python3 /opt/photoslam-benchmark/run.py monogs replica \
    --sequences all \
    --run-id replica-monogs-main \
    --continue-on-error
```

Replace `monogs` with `hislam2` in both positions for HI-SLAM2. For TANDEM,
record the quality and runtime presets separately:

```bash
./run_container.sh tandem \
  python3 /opt/photoslam-benchmark/run.py tandem replica \
    --sequences all \
    --run-id replica-tandem-dataset \
    --tandem-preset dataset \
    --continue-on-error

./run_container.sh tandem \
  python3 /opt/photoslam-benchmark/run.py tandem replica \
    --sequences all \
    --run-id replica-tandem-runtime \
    --tandem-preset runtime \
    --continue-on-error
```

`dataset` is the TANDEM geometry/tracking protocol; `runtime` enables its
runtime-oriented preloading and CUDA tracker. Results are written below
`/results/paper_benchmark/<run-id>`. Wrapper wall time and whole-device GPU
usage are kept separate from native metrics, and offline mesh export is not
included in system runtime.

The scripts never remove or replace existing images or containers. If a named
container already exists, they stop and report its name instead.
