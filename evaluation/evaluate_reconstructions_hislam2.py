#!/usr/bin/env python3
"""Batch Replica mesh alignment and HI-SLAM2 reconstruction evaluation.

The protocol matches HI-SLAM2's Replica evaluation:

1. Align each raw reconstruction to the GT frame with trajectory-based Sim(3).
2. Pass the aligned mesh to HI-SLAM2's scripts/eval_recon.py, which performs
   its own rigid ICP refinement and evaluates at a 5 cm threshold.

Add or remove entries in EXPERIMENTS to change the evaluated batch.
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
GT_TRAJECTORY = REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
GT_MESH = REPO_ROOT / "third_party/HI-SLAM2/data/Replica/gt_mesh/office0.ply"
SHARED_RECON_TRAJECTORY = REPLICA_RESULT_ROOT / "CameraTrajectory_TUM.txt"
HI_SLAM2_ROOT = REPO_ROOT / "third_party/HI-SLAM2"
HI_SLAM2_EVALUATOR = HI_SLAM2_ROOT / "scripts/eval_recon.py"
HI_SLAM2_PYTHON = Path.home() / "miniconda3/envs/hislam2/bin/python"
MESH_EVAL = REPO_ROOT / "bin/mesh_eval"
SUMMARY_DIR = REPLICA_RESULT_ROOT / "hi_slam2_reconstruction_batch"


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    directory: Path
    mesh: Path | None = None
    recon_trajectory: Path | None = None


EXPERIMENTS = [
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
        name="svraster_5801_shutdown",
        family="SVRaster",
        directory=REPLICA_RESULT_ROOT / "experiments/5801_shutdown",
    ),
    Experiment(
        name="svraster_6801_shutdown",
        family="SVRaster",
        directory=REPLICA_RESULT_ROOT / "experiments/6801_shutdown",
    ),
]


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


def resolve_recon_trajectory(experiment: Experiment) -> tuple[Path, bool]:
    if experiment.recon_trajectory is not None:
        return experiment.recon_trajectory.resolve(), False

    local_trajectory = experiment.directory / "CameraTrajectory_TUM.txt"
    if local_trajectory.is_file():
        return local_trajectory.resolve(), False

    return SHARED_RECON_TRAJECTORY.resolve(), True


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


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


def align_mesh(
    experiment: Experiment,
    mesh: Path,
    recon_trajectory: Path,
) -> Path:
    output_dir = experiment.directory / "hi_alignment"
    aligned_mesh = output_dir / "recon_mesh_aligned.ply"

    command = [
        str(MESH_EVAL),
        "--eval_mode=current",
        f"--recon={mesh}",
        f"--gt={GT_MESH}",
        f"--out={output_dir}",
        "--tau_cm=5.0",
        "--align_recon_to_gt=1",
        f"--traj={GT_TRAJECTORY}",
        "--traj_mode=c2w",
        f"--recon_traj_tum={recon_trajectory}",
        "--save_aligned_mesh=1",
        "--eval_floaters=0",
        "--eval_gaussian_support=0",
        "--eval_voxel_support=0",
        "--recon_samples=1000",
        "--gt_samples=1000",
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
    experiment: Experiment,
    aligned_mesh: Path,
) -> tuple[dict[str, float], Path]:
    result_path = experiment.directory / "eval_recon_hi_official.txt"

    command = [
        str(HI_SLAM2_PYTHON),
        str(HI_SLAM2_EVALUATOR),
        str(aligned_mesh),
        str(GT_MESH),
        "--eval_3d",
        "--save",
        str(result_path),
    ]
    run_logged(
        command,
        HI_SLAM2_ROOT,
        experiment.directory / "eval_recon_hi_official.log",
    )
    require_file(result_path, f"HI-SLAM2 metrics for {experiment.name}")
    metrics = parse_hi_metrics(result_path)
    return metrics, result_path


def result_from_metrics(
    experiment: Experiment,
    raw_mesh: Path,
    aligned_mesh: Path,
    recon_trajectory: Path,
    result_path: Path,
    metrics: dict[str, float],
) -> dict[str, Any]:
    accuracy = metrics["mean precision"]
    completeness = metrics["mean recall"]
    recall = metrics["recall"]
    return {
        "name": experiment.name,
        "family": experiment.family,
        "accuracy_m": accuracy,
        "completeness_m": completeness,
        "chamfer_l1_m": accuracy + completeness,
        "precision_5cm": metrics["precision"],
        "recall_5cm": recall,
        "completion_5cm": recall,
        "completion_5cm_percent": 100.0 * recall,
        "fscore_5cm": metrics["f-score"],
        "raw_mesh": str(raw_mesh),
        "aligned_mesh": str(aligned_mesh),
        "recon_trajectory": str(recon_trajectory),
        "evaluation_file": str(result_path),
    }


SUMMARY_FIELDS = [
    "name",
    "family",
    "accuracy_m",
    "completeness_m",
    "chamfer_l1_m",
    "precision_5cm",
    "recall_5cm",
    "completion_5cm",
    "completion_5cm_percent",
    "fscore_5cm",
    "raw_mesh",
    "aligned_mesh",
    "recon_trajectory",
    "evaluation_file",
]


def write_summaries(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "reconstruction_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / "reconstruction_summary.json"
    write_json(
        json_path,
        {
            "protocol": {
                "alignment": "trajectory Sim(3) via mesh_eval eval_mode=current",
                "refinement": "rigid ICP inside HI-SLAM2 eval_recon.py",
                "distance_threshold_m": 0.05,
                "gt_mesh": str(GT_MESH),
                "gt_trajectory": str(GT_TRAJECTORY),
            },
            "results": results,
        },
    )

    markdown_path = output_dir / "reconstruction_summary.md"
    lines = [
        "| Experiment | Family | Acc. m ↓ | Compl. m ↓ | Chamfer m ↓ | "
        "Prec@5 ↑ | Recall/Completion@5 ↑ | F@5 ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {name} | {family} | {accuracy_m:.6f} | "
            "{completeness_m:.6f} | {chamfer_l1_m:.6f} | "
            "{precision_5cm:.4f} | {completion_5cm:.4f} | "
            "{fscore_5cm:.4f} |".format(**result)
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved batch CSV:      {csv_path}")
    print(f"Saved batch JSON:     {json_path}")
    print(f"Saved batch Markdown: {markdown_path}")


def main() -> int:
    require_file(MESH_EVAL, "mesh_eval executable")
    require_file(GT_TRAJECTORY, "Replica GT trajectory")
    require_file(GT_MESH, "HI-SLAM2 Replica GT mesh")
    require_file(HI_SLAM2_EVALUATOR, "HI-SLAM2 reconstruction evaluator")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python interpreter")

    results: list[dict[str, Any]] = []
    shared_trajectory_users: list[str] = []
    for index, experiment in enumerate(EXPERIMENTS, start=1):
        print(
            f"\n[{index}/{len(EXPERIMENTS)}] Evaluating "
            f"{experiment.name} ({experiment.family})"
        )
        raw_mesh = discover_mesh(experiment)
        require_file(raw_mesh, f"Raw mesh for {experiment.name}")
        recon_trajectory, uses_shared_trajectory = resolve_recon_trajectory(
            experiment
        )
        require_file(
            recon_trajectory,
            f"Reconstruction trajectory for {experiment.name}",
        )
        if uses_shared_trajectory:
            shared_trajectory_users.append(experiment.name)
            print(
                f"[{experiment.name}] WARNING: no trajectory was saved "
                f"inside the experiment; using shared trajectory "
                f"{recon_trajectory}"
            )

        aligned_mesh = align_mesh(
            experiment,
            raw_mesh,
            recon_trajectory,
        )
        metrics, result_path = evaluate_aligned_mesh(
            experiment,
            aligned_mesh,
        )
        result = result_from_metrics(
            experiment,
            raw_mesh,
            aligned_mesh,
            recon_trajectory,
            result_path,
            metrics,
        )
        results.append(result)
        print(
            f"[{experiment.name}] Acc={result['accuracy_m']:.6f} m, "
            f"Compl={result['completeness_m']:.6f} m, "
            f"Completion@5={result['completion_5cm_percent']:.2f}%, "
            f"F@5={result['fscore_5cm']:.4f}"
        )

    if shared_trajectory_users:
        joined = ", ".join(shared_trajectory_users)
        print(
            "\nWARNING: These experiments used the shared "
            f"CameraTrajectory_TUM.txt: {joined}. For a scientifically exact "
            "comparison, place the matching trajectory in each experiment "
            "directory before rerunning."
        )

    write_summaries(results, SUMMARY_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
