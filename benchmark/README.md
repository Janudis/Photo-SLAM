# Paper benchmark launcher

`benchmark/run.py` runs the native methods currently built by this repository:

- `ours`: model-free rendered-depth densification;
- `ours_mvs`: CVA-MVSNet-assisted densification;
- `photoslam`: the original 3D Gaussian Photo-SLAM mapper.

The two voxel presets use the same dataset configuration and differ only in
the selected densification mode. Omnidata, MVS TSDF evidence, and Rerun are
disabled in both presets. The launcher validates the fixed base configuration,
ORB vocabulary, and MVS model by SHA-256 before starting.

Each trial gets a new directory below:

```text
<results-root>/paper_benchmark/<run-id>/<dataset>/<sequence>/<method>/trial_XX/
```

The directory contains the exact configuration snapshots, `run.log`,
`provenance.json`, and the normal numeric `*_shutdown` output. Existing run
directories are never overwritten.

## Server usage

Enter the already-created benchmark container from the host:

```bash
cd ~/workspaces/dimitris/Photo-SLAM/docker/ubuntu-jammy-blackwell
export PHOTOSLAM_CONTAINER=dimitris-photoslam-benchmark
./run_container.sh
```

Inside the container, validate the full Replica matrix without running it:

```bash
cd ~/workspaces/dimitris/Photo-SLAM
python3 benchmark/run.py replica \
  --sequences all \
  --methods all \
  --run-id replica-main \
  --dry-run
```

Run the matrix after the dry run succeeds:

```bash
python3 benchmark/run.py replica \
  --sequences all \
  --methods ours ours_mvs photoslam \
  --run-id replica-main
```

Run one configuration first as a smoke test:

```bash
python3 benchmark/run.py replica \
  --sequences office0 \
  --methods ours \
  --run-id replica-office0-smoke
```

Equivalent TUM and ScanNet commands are:

```bash
python3 benchmark/run.py tum \
  --sequences all \
  --methods all \
  --run-id tum-main

python3 benchmark/run.py scannet \
  --sequences scene0000_00 \
  --methods all \
  --run-id scannet-main
```

Use `--repetitions 3` when repeated trials are required. Jobs are deliberately
run sequentially so methods do not compete for GPU memory or compute.

## Scope

This launcher covers only binaries compiled in the Photo-SLAM container.
TANDEM, MonoGS, HI-SLAM2, GO-SLAM, NICER-SLAM, Splat-SLAM, and DROID-SLAM
require separate pinned environments because their Python, PyTorch, CUDA, and
extension requirements conflict. Their adapters should write into the same
directory contract and provenance format rather than being installed into the
Photo-SLAM image.
