#!/usr/bin/env python3
"""Batch Replica and ScanNet alignment and HI-SLAM2 reconstruction evaluation.

The protocol matches the HI-SLAM2 reconstruction evaluation:

1. Align each raw reconstruction to the GT frame with trajectory-based Sim(3).
2. Pass the aligned mesh to HI-SLAM2's scripts/eval_recon.py, which performs
   its own rigid ICP refinement and evaluates at the requested threshold.

Replica is evaluated independently against both its culled and original mesh.
Add or remove entries in a dataset's experiments tuple to change the batch.
Running this file directly uses HI-SLAM2's original 5 cm threshold.

# Original HI-SLAM2 5 cm evaluation
python3 evaluation/evaluate_reconstructions_hislam2.py
# Identical evaluation at 1 cm
python3 evaluation/evaluate_reconstructions_hislam2_1cm.py

"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLICA_RESULT_ROOT = REPO_ROOT / "results/replica_rgbd_voxel/office0"
REPLICA_PHOTO_SLAM_ROOT = REPO_ROOT / "results/replica_rgbd_original/office0"
REPLICA_GT_TRAJECTORY = REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
REPLICA_GT_MESH_CULLED = (
    REPO_ROOT / "third_party/ESLAM/cull_replica_mesh/office0_culled.ply"
)
REPLICA_GT_MESH_ORIGINAL = REPO_ROOT / "scripts/data/Replica/office0_mesh.ply"

SCANNET_RESULT_ROOT = REPO_ROOT / "results/scannet_rgbd_voxel/scene0000_00"
SCANNET_PHOTO_SLAM_ROOT = (
    REPO_ROOT / "results/scannet_rgbd_original/scene0000_00"
)
SCANNET_DATA_ROOT = REPO_ROOT / "scripts/data/ScanNet/scans/scene0000_00"

HI_SLAM2_ROOT = REPO_ROOT / "third_party/HI-SLAM2"
REPLICA_HI_SLAM2_ROOT = HI_SLAM2_ROOT / "outputs/replica/office0"
SCANNET_HI_SLAM2_ROOT = HI_SLAM2_ROOT / "outputs/scannet/scene0000_00"
HI_SLAM2_EVALUATOR = HI_SLAM2_ROOT / "scripts/eval_recon.py"
HI_SLAM2_PYTHON = Path.home() / "miniconda3/envs/hislam2/bin/python"
MESH_EVAL = REPO_ROOT / "bin/mesh_eval"
DEFAULT_DISTANCE_THRESHOLD_M = 0.05


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    directory: Path
    mesh: Path | None = None
    recon_trajectory: Path | None = None
    gt_trajectory: Path | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    gt_mesh: Path
    summary_dir: Path
    experiments: tuple[Experiment, ...]
    shared_recon_trajectory: Path
    artifact_tag: str = ""
    gt_trajectory: Path | None = None
    gt_pose_directory: Path | None = None


REPLICA_EXPERIMENTS = (
    Experiment(
        name="svrecon_1_shutdown",
        family="SVRecon",
        directory=REPLICA_RESULT_ROOT / "experiments_SVRECON/1_shutdown",
    ),
    Experiment(
        name="svrecon_2_shutdown",
        family="SVRecon",
        directory=REPLICA_RESULT_ROOT / "experiments_SVRECON/2_shutdown",
    ),
    Experiment(
        name="svrecon_3_shutdown",
        family="SVRecon",
        directory=REPLICA_RESULT_ROOT / "experiments_SVRECON/3_shutdown",
    ),
    Experiment(
        name="svrecon_4_shutdown",
        family="SVRecon",
        directory=REPLICA_RESULT_ROOT / "experiments_SVRECON/4_shutdown",
    ),
    # Experiment(
    #     name="svrecon_2376_shutdown",
    #     family="SVRecon",
    #     directory=REPLICA_RESULT_ROOT / "experiments_SVRECON/2376_shutdown",
    # ),
    Experiment(
        name="photoslam_3581_shutdown",
        family="Photo-SLAM",
        directory=REPLICA_PHOTO_SLAM_ROOT / "3581_shutdown",
        mesh=(
            REPLICA_PHOTO_SLAM_ROOT
            / "3581_shutdown/ply/point_cloud/iteration_3581"
            / "gaussian_surface_mesh.ply"
        ),
        recon_trajectory=REPLICA_PHOTO_SLAM_ROOT / "CameraTrajectory_TUM.txt",
    ),
    Experiment(
        name="hislam2_after_refinement",
        family="HI-SLAM2",
        directory=REPLICA_HI_SLAM2_ROOT,
        mesh=REPLICA_HI_SLAM2_ROOT / "tsdf_mesh_after_opt_w2.0.ply",
        recon_trajectory=REPLICA_HI_SLAM2_ROOT / "traj_full.txt",
    ),
)


SCANNET_EXPERIMENTS = (
    Experiment(
        name="svrecon_rgbd_fill_6961_shutdown",
        family="SVRecon",
        directory=SCANNET_RESULT_ROOT / "6961_shutdown",
    ),
    Experiment(
        name="svrecon_tsdf_evidence_5777_shutdown",
        family="SVRecon",
        directory=SCANNET_RESULT_ROOT / "5777_shutdown",
    ),
    Experiment(
        name="photoslam_22881_shutdown",
        family="Photo-SLAM",
        directory=SCANNET_PHOTO_SLAM_ROOT / "22881_shutdown",
        mesh=(
            SCANNET_PHOTO_SLAM_ROOT
            / "22881_shutdown/ply/point_cloud/iteration_22881"
            / "gaussian_surface_mesh.ply"
        ),
        recon_trajectory=SCANNET_PHOTO_SLAM_ROOT / "CameraTrajectory_TUM.txt",
    ),
    # Experiment(
    #     name="hislam2_after_refinement_w5",
    #     family="HI-SLAM2",
    #     directory=SCANNET_HI_SLAM2_ROOT,
    #     mesh=SCANNET_HI_SLAM2_ROOT / "tsdf_mesh_w5.0.ply",
    #     recon_trajectory=SCANNET_HI_SLAM2_ROOT / "traj_full.txt",
    # ),
)


DATASETS = (
    Dataset(
        name="Replica office0 (culled GT)",
        gt_mesh=REPLICA_GT_MESH_CULLED,
        gt_trajectory=REPLICA_GT_TRAJECTORY,
        shared_recon_trajectory=(
            REPLICA_RESULT_ROOT / "CameraTrajectory_TUM.txt"
        ),
        artifact_tag="culled",
        summary_dir=(
            REPLICA_RESULT_ROOT / "hi_slam2_reconstruction_batch_culled"
        ),
        experiments=REPLICA_EXPERIMENTS,
    ),
    Dataset(
        name="Replica office0 (original GT)",
        gt_mesh=REPLICA_GT_MESH_ORIGINAL,
        gt_trajectory=REPLICA_GT_TRAJECTORY,
        shared_recon_trajectory=(
            REPLICA_RESULT_ROOT / "CameraTrajectory_TUM.txt"
        ),
        artifact_tag="original",
        summary_dir=(
            REPLICA_RESULT_ROOT / "hi_slam2_reconstruction_batch_original"
        ),
        experiments=REPLICA_EXPERIMENTS,
    ),
    Dataset(
        name="ScanNet scene0000_00",
        gt_mesh=SCANNET_DATA_ROOT / "scene0000_00_vh_clean_2.ply",
        gt_pose_directory=SCANNET_DATA_ROOT / "pose",
        shared_recon_trajectory=(
            SCANNET_RESULT_ROOT / "CameraTrajectory_TUM.txt"
        ),
        summary_dir=(
            SCANNET_RESULT_ROOT / "hi_slam2_reconstruction_batch"
        ),
        experiments=SCANNET_EXPERIMENTS,
    ),
)


REQUIRED_HI_METRICS = {
    "mean precision",
    "mean recall",
    "precision",
    "recall",
    "f-score",
}

NUMPY_SCALAR_PATTERN = re.compile(
    r"\b(?:np|numpy)\.(?:float(?:16|32|64|128)?|"
    r"int(?:8|16|32|64)?|uint(?:8|16|32|64)?)\s*\("
)


def iteration_number(path: Path) -> int:
    match = re.fullmatch(r"iteration_(\d+)", path.parent.name)
    return int(match.group(1)) if match else -1


def discover_mesh(experiment: Experiment) -> Path:
    if experiment.mesh is not None:
        return experiment.mesh.resolve()

    model_root = experiment.directory / "ply/voxel_model"
    candidates = list(model_root.glob("iteration_*/voxel_surface_mesh.ply"))
    if not candidates:
        raise FileNotFoundError(
            f"No voxel_surface_mesh.ply found below {model_root}"
        )
    return max(candidates, key=iteration_number).resolve()


def resolve_recon_trajectory(
    dataset: Dataset,
    experiment: Experiment,
) -> tuple[Path, bool]:
    if experiment.recon_trajectory is not None:
        return experiment.recon_trajectory.resolve(), False

    local_trajectory = experiment.directory / "CameraTrajectory_TUM.txt"
    if local_trajectory.is_file():
        return local_trajectory.resolve(), False

    return dataset.shared_recon_trajectory.resolve(), True


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def materialize_gt_trajectory(dataset: Dataset) -> Path:
    if dataset.gt_trajectory is not None:
        require_file(dataset.gt_trajectory, f"{dataset.name} GT trajectory")
        return dataset.gt_trajectory.resolve()

    if dataset.gt_pose_directory is None:
        raise RuntimeError(f"{dataset.name} has no GT trajectory source")
    if not dataset.gt_pose_directory.is_dir():
        raise FileNotFoundError(
            f"{dataset.name} GT pose directory does not exist: "
            f"{dataset.gt_pose_directory}"
        )

    pose_paths = sorted(dataset.gt_pose_directory.glob("*.txt"))
    if not pose_paths:
        raise FileNotFoundError(
            f"No ScanNet pose files found in {dataset.gt_pose_directory}"
        )

    trajectory_path = dataset.summary_dir / "gt_trajectory_row_major.txt"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pose_path in pose_paths:
        try:
            values = [
                float(token)
                for token in pose_path.read_text(encoding="utf-8").split()
            ]
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot parse GT pose {pose_path}") from error
        if len(values) != 16 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"GT pose must contain 16 finite values: {pose_path}"
            )
        lines.append(" ".join(f"{value:.12g}" for value in values))

    trajectory_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return trajectory_path.resolve()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_command(command: list[str], cwd: Path) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[cwd={cwd}] $ {rendered}", flush=True)


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    print_command(command, cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}; see {log_path}"
        )


def threshold_label(distance_threshold_m: float) -> str:
    threshold_cm = 100.0 * distance_threshold_m
    rounded_cm = round(threshold_cm)
    if math.isclose(threshold_cm, rounded_cm, abs_tol=1.0e-9):
        return f"{rounded_cm:d}cm"
    return f"{threshold_cm:g}cm".replace(".", "p")


def threshold_output_suffix(distance_threshold_m: float) -> str:
    if math.isclose(
        distance_threshold_m,
        DEFAULT_DISTANCE_THRESHOLD_M,
        abs_tol=1.0e-12,
    ):
        return ""
    return f"_{threshold_label(distance_threshold_m)}"


def summary_output_dir(
    dataset: Dataset,
    distance_threshold_m: float,
) -> Path:
    suffix = threshold_output_suffix(distance_threshold_m)
    if not suffix:
        return dataset.summary_dir
    return dataset.summary_dir.with_name(dataset.summary_dir.name + suffix)


def align_mesh(
    dataset: Dataset,
    experiment: Experiment,
    mesh: Path,
    recon_trajectory: Path,
    gt_trajectory: Path,
    distance_threshold_m: float,
) -> Path:
    alignment_dir_name = (
        "hi_alignment" + threshold_output_suffix(distance_threshold_m)
    )
    if dataset.artifact_tag:
        alignment_dir_name += f"_{dataset.artifact_tag}"
    output_dir = experiment.directory / alignment_dir_name
    aligned_mesh = output_dir / "recon_mesh_aligned.ply"

    command = [
        str(MESH_EVAL),
        "--eval_mode=current",
        f"--recon={mesh}",
        f"--gt={dataset.gt_mesh}",
        f"--out={output_dir}",
        f"--tau_cm={100.0 * distance_threshold_m:g}",
        "--align_recon_to_gt=1",
        f"--traj={gt_trajectory}",
        "--traj_mode=c2w",
        f"--recon_traj_tum={recon_trajectory}",
        "--save_aligned_mesh=1",
        "--alignment_only=1",
        "--eval_floaters=0",
        "--eval_gaussian_support=0",
        "--eval_voxel_support=0",
    ]
    run_logged(command, REPO_ROOT, output_dir / "alignment.log")
    require_file(aligned_mesh, f"Aligned mesh for {experiment.name}")
    return aligned_mesh.resolve()


def parse_hi_metrics(path: Path) -> dict[str, float]:
    try:
        serialized = path.read_text(encoding="utf-8")
        serialized = NUMPY_SCALAR_PATTERN.sub("(", serialized)
        value = ast.literal_eval(serialized)
    except (OSError, SyntaxError, ValueError) as error:
        raise RuntimeError(f"Cannot parse HI-SLAM2 metrics from {path}") from error

    if not isinstance(value, dict):
        raise RuntimeError(f"HI-SLAM2 output is not a dictionary: {path}")
    missing = REQUIRED_HI_METRICS.difference(value)
    if missing:
        raise RuntimeError(
            f"HI-SLAM2 output is missing {sorted(missing)}: {path}"
        )

    metrics = {key: float(value[key]) for key in REQUIRED_HI_METRICS}
    if not all(math.isfinite(metric) for metric in metrics.values()):
        raise RuntimeError(f"HI-SLAM2 output contains non-finite values: {path}")
    return metrics


def evaluate_aligned_mesh(
    dataset: Dataset,
    experiment: Experiment,
    aligned_mesh: Path,
    distance_threshold_m: float,
) -> tuple[dict[str, float], Path]:
    result_filename = (
        "eval_recon_hi_official"
        + threshold_output_suffix(distance_threshold_m)
    )
    if dataset.artifact_tag:
        result_filename += f"_{dataset.artifact_tag}"
    result_path = experiment.directory / f"{result_filename}.txt"

    command = [
        str(HI_SLAM2_PYTHON),
        str(HI_SLAM2_EVALUATOR),
        str(aligned_mesh),
        str(dataset.gt_mesh),
        "--eval_3d",
        "--distance_thresh",
        f"{distance_threshold_m:.12g}",
        "--save",
        str(result_path),
    ]
    run_logged(
        command,
        HI_SLAM2_ROOT,
        experiment.directory / f"{result_filename}.log",
    )
    require_file(result_path, f"HI-SLAM2 metrics for {experiment.name}")
    metrics = parse_hi_metrics(result_path)
    return metrics, result_path


def result_from_metrics(
    dataset: Dataset,
    experiment: Experiment,
    raw_mesh: Path,
    aligned_mesh: Path,
    recon_trajectory: Path,
    result_path: Path,
    metrics: dict[str, float],
    distance_threshold_m: float,
) -> dict[str, Any]:
    accuracy = metrics["mean precision"]
    completeness = metrics["mean recall"]
    recall = metrics["recall"]
    metric_suffix = threshold_label(distance_threshold_m)
    return {
        "dataset": dataset.name,
        "name": experiment.name,
        "family": experiment.family,
        "accuracy_m": accuracy,
        "completeness_m": completeness,
        "chamfer_l1_m": accuracy + completeness,
        f"precision_{metric_suffix}": metrics["precision"],
        f"recall_{metric_suffix}": recall,
        f"completion_{metric_suffix}": recall,
        f"completion_{metric_suffix}_percent": 100.0 * recall,
        f"fscore_{metric_suffix}": metrics["f-score"],
        "raw_mesh": str(raw_mesh),
        "aligned_mesh": str(aligned_mesh),
        "recon_trajectory": str(recon_trajectory),
        "evaluation_file": str(result_path),
    }


def summary_fields(distance_threshold_m: float) -> list[str]:
    metric_suffix = threshold_label(distance_threshold_m)
    return [
        "dataset",
        "name",
        "family",
        "accuracy_m",
        "completeness_m",
        "chamfer_l1_m",
        f"precision_{metric_suffix}",
        f"recall_{metric_suffix}",
        f"completion_{metric_suffix}",
        f"completion_{metric_suffix}_percent",
        f"fscore_{metric_suffix}",
        "raw_mesh",
        "aligned_mesh",
        "recon_trajectory",
        "evaluation_file",
    ]


def write_summaries(
    dataset: Dataset,
    gt_trajectory: Path,
    results: list[dict[str, Any]],
    distance_threshold_m: float,
) -> None:
    output_dir = summary_output_dir(dataset, distance_threshold_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_suffix = threshold_label(distance_threshold_m)
    precision_key = f"precision_{metric_suffix}"
    completion_key = f"completion_{metric_suffix}"
    fscore_key = f"fscore_{metric_suffix}"

    csv_path = output_dir / "reconstruction_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_fields(distance_threshold_m),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "reconstruction_summary.json"
    write_json(
        json_path,
        {
            "protocol": {
                "alignment": (
                    "trajectory Sim(3) via mesh_eval eval_mode=current "
                    "alignment_only"
                ),
                "refinement": "rigid ICP inside HI-SLAM2 eval_recon.py",
                "distance_threshold_m": distance_threshold_m,
                "dataset": dataset.name,
                "gt_mesh": str(dataset.gt_mesh),
                "gt_trajectory": str(gt_trajectory),
            },
            "results": results,
        },
    )

    markdown_path = output_dir / "reconstruction_summary.md"
    lines = [
        "| Experiment | Family | Acc. m ↓ | Compl. m ↓ | Chamfer m ↓ | "
        f"Prec@{metric_suffix} ↑ | Recall/Completion@{metric_suffix} ↑ | "
        f"F@{metric_suffix} ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['family']} | "
            f"{result['accuracy_m']:.6f} | "
            f"{result['completeness_m']:.6f} | "
            f"{result['chamfer_l1_m']:.6f} | "
            f"{result[precision_key]:.4f} | "
            f"{result[completion_key]:.4f} | "
            f"{result[fscore_key]:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved batch CSV:      {csv_path}")
    print(f"Saved batch JSON:     {json_path}")
    print(f"Saved batch Markdown: {markdown_path}")


def main(
    distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M,
) -> int:
    if not math.isfinite(distance_threshold_m) or distance_threshold_m <= 0.0:
        raise ValueError(
            "distance_threshold_m must be a positive finite value"
        )

    require_file(MESH_EVAL, "mesh_eval executable")
    require_file(HI_SLAM2_EVALUATOR, "HI-SLAM2 reconstruction evaluator")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python interpreter")

    for dataset_index, dataset in enumerate(DATASETS, start=1):
        print(
            f"\n=== Dataset {dataset_index}/{len(DATASETS)}: "
            f"{dataset.name} ==="
        )
        require_file(dataset.gt_mesh, f"{dataset.name} GT mesh")
        gt_trajectory = materialize_gt_trajectory(dataset)

        results: list[dict[str, Any]] = []
        shared_trajectory_users: list[str] = []
        for index, experiment in enumerate(dataset.experiments, start=1):
            print(
                f"\n[{index}/{len(dataset.experiments)}] Evaluating "
                f"{experiment.name} ({experiment.family})"
            )
            raw_mesh = discover_mesh(experiment)
            require_file(raw_mesh, f"Raw mesh for {experiment.name}")
            recon_trajectory, uses_shared_trajectory = (
                resolve_recon_trajectory(dataset, experiment)
            )
            require_file(
                recon_trajectory,
                f"Reconstruction trajectory for {experiment.name}",
            )
            if uses_shared_trajectory:
                shared_trajectory_users.append(experiment.name)
                print(
                    f"[{experiment.name}] WARNING: no run-local trajectory; "
                    f"using legacy shared trajectory {recon_trajectory}"
                )

            alignment_gt_trajectory = (
                experiment.gt_trajectory.resolve()
                if experiment.gt_trajectory is not None
                else gt_trajectory
            )
            require_file(
                alignment_gt_trajectory,
                f"Alignment GT trajectory for {experiment.name}",
            )
            aligned_mesh = align_mesh(
                dataset,
                experiment,
                raw_mesh,
                recon_trajectory,
                alignment_gt_trajectory,
                distance_threshold_m,
            )
            metrics, result_path = evaluate_aligned_mesh(
                dataset,
                experiment,
                aligned_mesh,
                distance_threshold_m,
            )
            result = result_from_metrics(
                dataset,
                experiment,
                raw_mesh,
                aligned_mesh,
                recon_trajectory,
                result_path,
                metrics,
                distance_threshold_m,
            )
            results.append(result)
            metric_suffix = threshold_label(distance_threshold_m)
            print(
                f"[{experiment.name}] Acc={result['accuracy_m']:.6f} m, "
                f"Compl={result['completeness_m']:.6f} m, "
                f"Completion@{metric_suffix}="
                f"{result[f'completion_{metric_suffix}_percent']:.2f}%, "
                f"F@{metric_suffix}="
                f"{result[f'fscore_{metric_suffix}']:.4f}"
            )

        if shared_trajectory_users:
            print(
                "\nLegacy shared trajectory used by: "
                + ", ".join(shared_trajectory_users)
            )

        write_summaries(
            dataset,
            gt_trajectory,
            results,
            distance_threshold_m,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
