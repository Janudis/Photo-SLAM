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

After the complete 24-job Replica matrix finishes, evaluate all three native
methods with one command:

```bash
python3 benchmark/evaluate.py replica --run-id replica-main
```

The evaluator uses `evo_ape` with Sim(3) alignment for tracking, aligns each
mesh through `mesh_eval`, and then invokes the official HI-SLAM2 3D
reconstruction evaluator at 5 cm. PSNR, SSIM, and AlexNet LPIPS are averaged
over all keyframes rendered by each completed mapping run; the evaluated frame
IDs and counts are stored with the results.
Its consolidated output is written to:

```text
<results-root>/paper_benchmark/<run-id>/evaluation/replica_summary.json
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

## Full Replica MVS-TSDF configurations

After selecting the pruning behavior on `office0`, run the fixed MVS-TSDF
configuration over all eight Replica scenes with two explicit presets:

- `ours_mvs_tsdf_geometry` enables MVS-consistency pruning;
- `ours_mvs_tsdf_rendering` uses scheduled SDF/near/far pruning only.

Both final MVS-TSDF presets use a 256 m fixed octree root. With
`Model.outside_level=5`, this provides 8 m of initial inside extent for all
Replica scenes while retaining the configured insertion voxel size.

Both presets use full-image MVS-TSDF densification with a two-voxel truncation
band and MVS depth supervision with weight `0.001`. Direct rendered-depth and
direct MVS hole insertion, co-visibility pruning, final refinement, Omnidata,
and Rerun are disabled. The presets differ only in MVS-consistency pruning.
They are deliberately excluded from `--methods all` and must be named:

```bash
python3 benchmark/run.py replica \
  --sequences all \
  --methods ours_mvs_tsdf_geometry ours_mvs_tsdf_rendering \
  --run-id replica-mvs-tsdf-final-v1
```

Evaluate the resulting 16-job matrix with the same consolidated Replica
protocol used by the main benchmark:

```bash
DEPS=/tmp/photoslam-benchmark-eval
PYTHONPATH="$DEPS" PATH="$DEPS/bin:$PATH" \
python3 benchmark/evaluate.py replica \
  --run-id replica-mvs-tsdf-final-v1 \
  --methods ours_mvs_tsdf_geometry ours_mvs_tsdf_rendering
```

Use the geometry preset for the reconstruction table and the rendering preset
for the appearance table. Report them as distinct configurations rather than
combining their best metrics into one row.

## Replica MVS-TSDF pruning ablation

`ablate_densification.py` runs only the monocular SVRecon mapper on Replica
`office0`. Every variant uses full-image MVS-TSDF densification, a two-voxel
TSDF truncation band, and MVS depth regularization with weight `0.001`. Direct
rendered-depth and direct MVS hole insertion remain disabled. The variants
change only pruning behavior, leaving the MVS depth and densification settings
fixed.

This is intentionally different from the `ours_mvs` method used by the main
Replica table. That benchmark enabled direct MVS insertion at rendered holes,
disabled MVS-TSDF evidence, and used zero weights for learned-depth and normal
supervision. Its pruning used co-visibility and final refinement, while MVS
consistency pruning was disabled. The `surface_views_final` ablation below
reuses those pruning switches, but retains the fixed MVS-TSDF and depth-loss
settings of this study.

The six configurations compare scheduled SVRecon SDF/near/far pruning,
renderer co-visibility, MVS support/free-space consistency, the shutdown
refinement switches used by the table experiments, and the combined online
pruning methods with and without final refinement.

Inside the benchmark container, inspect the complete matrix without creating
outputs:

```bash
python3 benchmark/ablate_densification.py \
  --run-id replica-office0-mvs-tsdf-pruning-v1 \
  --dry-run
```

Run all configurations and immediately evaluate reconstruction and rendering:

```bash
DEPS=/tmp/photoslam-benchmark-eval
PYTHONPATH="$DEPS" PATH="$DEPS/bin:$PATH" \
python3 benchmark/ablate_densification.py \
  --run-id replica-office0-mvs-tsdf-pruning-v1
```

The default `core` suite runs all six configurations. For a paper ablation,
`--repetitions 3` reports the mean and population standard deviation across
independent SLAM runs. If evaluation is interrupted after mapping has completed,
rerun only that stage with:

```bash
DEPS=/tmp/photoslam-benchmark-eval
PYTHONPATH="$DEPS" PATH="$DEPS/bin:$PATH" \
python3 benchmark/ablate_densification.py \
  --run-id replica-office0-mvs-tsdf-pruning-v1 \
  --evaluate-only
```

Each trial retains its effective YAML, run log, normal shutdown output, and
evaluation cache. The single consolidated result is:

```text
<results-root>/densification_ablation/<run-id>/evaluation/summary.json
```

## Scope

This launcher covers only binaries compiled in the Photo-SLAM container.
TANDEM, MonoGS, HI-SLAM2, GO-SLAM, NICER-SLAM, Splat-SLAM, and DROID-SLAM
require separate pinned environments because their Python, PyTorch, CUDA, and
extension requirements conflict. Their adapters should write into the same
directory contract and provenance format rather than being installed into the
Photo-SLAM image.
