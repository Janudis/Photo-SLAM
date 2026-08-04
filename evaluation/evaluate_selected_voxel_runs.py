#!/usr/bin/env python3
"""Evaluate the selected TUM, Replica, and ScanNet voxel runs.

The script always recomputes reconstruction alignment/evaluation and writes
combined rendering, computational, and reconstruction tables below
results/selected_voxel_runs_evaluation.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_reconstructions_hislam2 import (  # noqa: E402
    Dataset,
    Experiment,
    HI_SLAM2_EVALUATOR,
    HI_SLAM2_PYTHON,
    MESH_EVAL,
    align_mesh,
    evaluate_aligned_mesh,
    materialize_gt_trajectory,
    require_file,
    result_from_metrics,
    threshold_label,
    write_summaries,
)


DISTANCE_THRESHOLD_M = 0.05
OUTPUT_DIR = REPO_ROOT / "results/selected_voxel_runs_evaluation"


@dataclass(frozen=True)
class Run:
    key: str
    dataset: str
    configuration: str
    directory: Path
    evaluate_reconstruction: bool


RUNS = (
    Run(
        key="tum_3841_shutdown",
        dataset="TUM fr1/desk",
        configuration="SVRecon monocular + TANDEM MVS",
        directory=(
            REPO_ROOT
            / "results/tum_voxel/rgbd_dataset_freiburg1_desk/3841_shutdown"
        ),
        evaluate_reconstruction=False,
    ),
    Run(
        key="replica_5801_shutdown",
        dataset="Replica office0",
        configuration="SVRecon monocular + TANDEM MVS",
        directory=(
            REPO_ROOT / "results/replica_voxel/office0/5801_shutdown"
        ),
        evaluate_reconstruction=True,
    ),
    Run(
        key="scannet_6962_shutdown",
        dataset="ScanNet scene0000_00",
        configuration="SVRecon RGB-D + TSDF evidence",
        directory=(
            REPO_ROOT
            / "results/scannet_rgbd_voxel/scene0000_00/6962_shutdown"
        ),
        evaluate_reconstruction=True,
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def finite_float(value: Any, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {field} in {path}: {value!r}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {field} in {path}")
    return result


def parse_keyed_metric(path: Path) -> dict[int, float]:
    require_file(path, "per-keyframe metric")
    values: dict[int, float] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise RuntimeError(f"Expected two columns at {path}:{line_number}")
        try:
            keyframe_id = int(tokens[0])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe ID at {path}:{line_number}"
            ) from error
        value = finite_float(tokens[1], "metric", path)
        if keyframe_id in values:
            raise RuntimeError(f"Duplicate keyframe {keyframe_id} in {path}")
        values[keyframe_id] = value
    if not values:
        raise RuntimeError(f"No metrics found in {path}")
    return values


def latest_surface_mesh(run: Run) -> Path:
    candidates = list(
        (run.directory / "ply/voxel_model").glob(
            "iteration_*/voxel_surface_mesh.ply"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one voxel surface mesh for {run.key}; "
            f"found {len(candidates)}"
        )
    require_file(candidates[0], f"surface mesh for {run.key}")
    return candidates[0].resolve()


def collect_rendering(run: Run) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    psnr = parse_keyed_metric(run.directory / "psnr.txt")
    ssim = parse_keyed_metric(run.directory / "dssim.txt")
    if set(psnr) != set(ssim):
        raise RuntimeError(f"PSNR/SSIM keyframes differ for {run.key}")

    per_keyframe = [
        {
            "dataset": run.dataset,
            "experiment": run.key,
            "keyframe_id": keyframe_id,
            "psnr": psnr[keyframe_id],
            "ssim": ssim[keyframe_id],
        }
        for keyframe_id in sorted(psnr)
    ]
    return (
        {
            "dataset": run.dataset,
            "experiment": run.key,
            "configuration": run.configuration,
            "evaluated_keyframes": len(per_keyframe),
            "psnr": statistics.fmean(row["psnr"] for row in per_keyframe),
            "ssim": statistics.fmean(row["ssim"] for row in per_keyframe),
        },
        per_keyframe,
    )


def collect_computational(run: Run) -> dict[str, Any]:
    path = run.directory / "runtime_metrics.json"
    runtime = load_json(path)
    frames = int(runtime.get("frames", 0))
    seconds = finite_float(runtime.get("total_seconds"), "total_seconds", path)
    if frames <= 0 or seconds <= 0.0:
        raise RuntimeError(f"Invalid frame count or runtime in {path}")
    return {
        "dataset": run.dataset,
        "experiment": run.key,
        "configuration": run.configuration,
        "frames": frames,
        "runtime_seconds": seconds,
        "fps": frames / seconds,
        "gpu_peak_allocated_mb": finite_float(
            runtime.get("gpu_memory_allocated_mb"),
            "gpu_memory_allocated_mb",
            path,
        ),
        "gpu_peak_reserved_mb": finite_float(
            runtime.get("gpu_memory_reserved_mb"),
            "gpu_memory_reserved_mb",
            path,
        ),
        "runtime_scope": str(runtime.get("runtime_scope", "")),
    }


def reconstruction_datasets() -> tuple[Dataset, ...]:
    replica = next(run for run in RUNS if run.key == "replica_5801_shutdown")
    scannet = next(run for run in RUNS if run.key == "scannet_6962_shutdown")
    replica_experiment = Experiment(
        name=replica.key,
        family=replica.configuration,
        directory=replica.directory,
        mesh=latest_surface_mesh(replica),
        recon_trajectory=replica.directory / "CameraTrajectory_TUM.txt",
    )
    scannet_experiment = Experiment(
        name=scannet.key,
        family=scannet.configuration,
        directory=scannet.directory,
        mesh=latest_surface_mesh(scannet),
        recon_trajectory=scannet.directory / "CameraTrajectory_TUM.txt",
    )
    replica_gt_trajectory = REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
    return (
        Dataset(
            name="Replica office0 (culled GT)",
            gt_mesh=(
                REPO_ROOT
                / "third_party/ESLAM/cull_replica_mesh/office0_culled.ply"
            ),
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=replica_experiment.recon_trajectory,
            artifact_tag="culled",
            summary_dir=OUTPUT_DIR / "reconstruction/replica_culled",
            experiments=(replica_experiment,),
        ),
        Dataset(
            name="Replica office0 (original GT)",
            gt_mesh=REPO_ROOT / "scripts/data/Replica/office0_mesh.ply",
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=replica_experiment.recon_trajectory,
            artifact_tag="original",
            summary_dir=OUTPUT_DIR / "reconstruction/replica_original",
            experiments=(replica_experiment,),
        ),
        Dataset(
            name="ScanNet scene0000_00",
            gt_mesh=(
                REPO_ROOT
                / "scripts/data/ScanNet/scans/scene0000_00"
                / "scene0000_00_vh_clean_2.ply"
            ),
            gt_pose_directory=(
                REPO_ROOT
                / "scripts/data/ScanNet/scans/scene0000_00/pose"
            ),
            shared_recon_trajectory=scannet_experiment.recon_trajectory,
            summary_dir=OUTPUT_DIR / "reconstruction/scannet",
            experiments=(scannet_experiment,),
        ),
    )


def evaluate_reconstruction() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset in reconstruction_datasets():
        print(f"\n=== Reconstruction: {dataset.name} ===")
        require_file(dataset.gt_mesh, f"GT mesh for {dataset.name}")
        gt_trajectory = materialize_gt_trajectory(dataset)
        dataset_results: list[dict[str, Any]] = []
        for experiment in dataset.experiments:
            assert experiment.mesh is not None
            assert experiment.recon_trajectory is not None
            require_file(experiment.recon_trajectory, "reconstruction trajectory")
            aligned_mesh = align_mesh(
                dataset,
                experiment,
                experiment.mesh,
                experiment.recon_trajectory,
                gt_trajectory,
                DISTANCE_THRESHOLD_M,
            )
            metrics, result_path = evaluate_aligned_mesh(
                dataset,
                experiment,
                aligned_mesh,
                DISTANCE_THRESHOLD_M,
            )
            result = result_from_metrics(
                dataset,
                experiment,
                experiment.mesh,
                aligned_mesh,
                experiment.recon_trajectory,
                result_path,
                metrics,
                DISTANCE_THRESHOLD_M,
            )
            dataset_results.append(result)
            results.append(result)
        write_summaries(
            dataset,
            gt_trajectory,
            dataset_results,
            DISTANCE_THRESHOLD_M,
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, protocol: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"protocol": protocol, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def rendering_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Experiment | Keyframes | PSNR ↑ | SSIM ↑ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['experiment']} | "
            f"{row['evaluated_keyframes']} | {row['psnr']:.4f} | "
            f"{row['ssim']:.4f} |"
        )
    return "\n".join(lines)


def computational_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Experiment | Runtime s ↓ | FPS ↑ | GPU reserved MB ↓ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['experiment']} | "
            f"{row['runtime_seconds']:.3f} | {row['fps']:.2f} | "
            f"{row['gpu_peak_reserved_mb']:.2f} |"
        )
    return "\n".join(lines)


def reconstruction_table(rows: list[dict[str, Any]]) -> str:
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    lines = [
        "| Dataset | Experiment | Acc. m ↓ | Compl. m ↓ | Chamfer m ↓ | "
        f"Prec@{suffix} ↑ | Recall@{suffix} ↑ | Completion@{suffix} ↑ | "
        f"F@{suffix} ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['name']} | "
            f"{row['accuracy_m']:.6f} | {row['completeness_m']:.6f} | "
            f"{row['chamfer_l1_m']:.6f} | "
            f"{row[f'precision_{suffix}']:.4f} | "
            f"{row[f'recall_{suffix}']:.4f} | "
            f"{row[f'completion_{suffix}_percent']:.2f}% | "
            f"{row[f'fscore_{suffix}']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    require_file(MESH_EVAL, "mesh_eval executable")
    require_file(HI_SLAM2_EVALUATOR, "HI-SLAM2 evaluator")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python")
    for run in RUNS:
        if not run.directory.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run.directory}")

    rendering: list[dict[str, Any]] = []
    rendering_per_keyframe: list[dict[str, Any]] = []
    for run in RUNS:
        summary, per_keyframe = collect_rendering(run)
        rendering.append(summary)
        rendering_per_keyframe.extend(per_keyframe)
    computational = [collect_computational(run) for run in RUNS]
    reconstruction = evaluate_reconstruction()

    render_md = rendering_table(rendering)
    compute_md = computational_table(computational)
    reconstruction_md = reconstruction_table(reconstruction)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "rendering_metrics.csv", rendering)
    write_csv(
        OUTPUT_DIR / "rendering_metrics_per_keyframe.csv",
        rendering_per_keyframe,
    )
    write_json(
        OUTPUT_DIR / "rendering_metrics.json",
        {
            "aggregation": "arithmetic mean over every saved keyframe render",
            "ssim_source": "dssim.txt stores SSIM despite its legacy filename",
        },
        rendering,
    )
    (OUTPUT_DIR / "rendering_metrics.md").write_text(
        render_md + "\n", encoding="utf-8"
    )

    write_csv(OUTPUT_DIR / "computational_metrics.csv", computational)
    write_json(
        OUTPUT_DIR / "computational_metrics.json",
        {
            "fps": "dataset frames / mapping total_seconds",
            "runtime_scope": (
                "mapping and tail optimization; shutdown evaluation rendering "
                "and mesh export excluded"
            ),
            "gpu_memory": (
                "peak PyTorch CUDA caching-allocator reserved memory captured "
                "at the end of the runtime scope"
            ),
        },
        computational,
    )
    (OUTPUT_DIR / "computational_metrics.md").write_text(
        compute_md + "\n", encoding="utf-8"
    )

    write_csv(OUTPUT_DIR / "reconstruction_metrics.csv", reconstruction)
    write_json(
        OUTPUT_DIR / "reconstruction_metrics.json",
        {
            "alignment": "trajectory Sim(3), followed by HI-SLAM2 rigid ICP",
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "completion_rate": "identical to recall, reported explicitly as percent",
            "tum": "omitted because no GT reconstruction mesh is provided",
        },
        reconstruction,
    )
    (OUTPUT_DIR / "reconstruction_metrics.md").write_text(
        reconstruction_md + "\n", encoding="utf-8"
    )

    summary = (
        "# Selected Voxel Run Evaluation\n\n"
        "## Rendering Quality\n\n"
        f"{render_md}\n\n"
        "## Computational Performance\n\n"
        f"{compute_md}\n\n"
        "## Reconstruction\n\n"
        f"{reconstruction_md}\n"
    )
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\nSaved evaluation to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
