#!/usr/bin/env python3
"""Evaluate monocular densification runs and the Photo-SLAM/HI-SLAM2/TANDEM baselines."""

from __future__ import annotations

import csv
import json
import math
import re
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
    run_logged,
    threshold_label,
    write_summaries,
)


DISTANCE_THRESHOLD_M = 0.05
OUTPUT_DIR = (
    REPO_ROOT / "results/monocular_densification_ablation_evaluation"
)
REPLICA_COMPLETE_RUN_LABELS = {
    "replica_rendered_depth": "Ours",
    "replica_mvs": "Ours + MVS",
    "replica_photoslam": "Photo-SLAM",
    "replica_hislam2": "HI-SLAM2 (3D GS)",
    "replica_tandem": "TANDEM (SDF)",
}
REPLICA_PRESENTATION_RUN_ORDER = (
    "replica_rendered_depth",
    "replica_mvs",
    "replica_photoslam",
    "replica_hislam2",
    "replica_tandem",
)
REPLICA_TABLE3_LABELS = {
    **REPLICA_COMPLETE_RUN_LABELS,
    "replica_photoslam": "Photo-SLAM (3D GS)",
}
HI_SLAM2_REPLICA_PHOTOMETRIC = REPO_ROOT / (
    "third_party/HI-SLAM2/outputs/replica/office0/psnr/"
    "after_opt/final_result.json"
)
REPLICA_DEPTH_INTRINSICS = {
    "img_w": 1200,
    "img_h": 680,
    "fx": 600.0,
    "fy": 600.0,
    "cx": 599.5,
    "cy": 339.5,
}
REPLICA_DEPTH_MAX_FRAMES = 1000
REPLICA_SURFACE_SAMPLES = 500_000
REPLICA_FLOATER_SAMPLES = 500_000
REPLICA_SUPPORT_SAMPLES_PER_PRIMITIVE = 32
FLOAT_TOKEN_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


@dataclass(frozen=True)
class Run:
    key: str
    method: str
    dataset_key: str
    dataset: str
    frame_count: int
    directory: Path
    evaluate_reconstruction: bool
    expected_rendered_depth: int
    expected_mvs: int
    expected_omnidata: int
    representation: str = "svrecon"
    validate_densification_config: bool = True
    family: str = "SVRecon monocular"
    native_map_override: Path | None = None
    surface_mesh_override: Path | None = None
    trajectory_override: Path | None = None
    alignment_gt_trajectory_override: Path | None = None


RUNS = (
    Run(
        key="tum_rendered_depth",
        method="Rendered-depth densification",
        dataset_key="tum",
        dataset="TUM fr1/desk",
        frame_count=613,
        directory=REPO_ROOT / (
            "results/tum_voxel/rgbd_dataset_freiburg1_desk/"
            "new_experiments/4641_shutdown"
        ),
        evaluate_reconstruction=False,
        expected_rendered_depth=1,
        expected_mvs=0,
        expected_omnidata=0,
    ),
    Run(
        key="tum_mvs",
        method="TANDEM MVS",
        dataset_key="tum",
        dataset="TUM fr1/desk",
        frame_count=613,
        directory=REPO_ROOT / (
            "results/tum_voxel/rgbd_dataset_freiburg1_desk/"
            "new_experiments/3841_shutdown_MVS"
        ),
        evaluate_reconstruction=False,
        expected_rendered_depth=0,
        expected_mvs=1,
        expected_omnidata=0,
    ),
    Run(
        key="tum_omnidata",
        method="Omnidata",
        dataset_key="tum",
        dataset="TUM fr1/desk",
        frame_count=613,
        directory=REPO_ROOT / (
            "results/tum_voxel/rgbd_dataset_freiburg1_desk/"
            "new_experiments/3841_shutdown_Omni"
        ),
        evaluate_reconstruction=False,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=1,
    ),
    Run(
        key="replica_rendered_depth",
        method="Rendered-depth densification",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_voxel/office0/new_experiments/6810_shutdown"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=1,
        expected_mvs=0,
        expected_omnidata=0,
    ),
    Run(
        key="replica_mvs",
        method="TANDEM MVS",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_voxel/office0/"
            "new_experiments/5801_shutdown_MVS"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=1,
        expected_omnidata=0,
    ),
    Run(
        key="replica_omnidata",
        method="Omnidata",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_voxel/office0/"
            "new_experiments/5848_shutdown_Omni"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=1,
    ),
    Run(
        key="replica_photoslam",
        method="Photo-SLAM",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_rgb_original/office0/6381_shutdown"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=0,
        representation="gaussian",
        validate_densification_config=False,
        family="Photo-SLAM monocular",
    ),
    Run(
        key="scannet_rendered_depth",
        method="Rendered-depth densification",
        dataset_key="scannet",
        dataset="ScanNet scene0000_00",
        frame_count=5578,
        directory=REPO_ROOT / (
            "results/scannet_voxel/scene0000_00/"
            "new_experiments/18961_shutdown"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=1,
        expected_mvs=0,
        expected_omnidata=0,
    ),
    Run(
        key="scannet_mvs",
        method="TANDEM MVS",
        dataset_key="scannet",
        dataset="ScanNet scene0000_00",
        frame_count=5578,
        directory=REPO_ROOT / (
            "results/scannet_voxel/scene0000_00/"
            "new_experiments/9361_shutdown_MVS"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=1,
        expected_omnidata=0,
    ),
    Run(
        key="scannet_omnidata",
        method="Omnidata",
        dataset_key="scannet",
        dataset="ScanNet scene0000_00",
        frame_count=5578,
        directory=REPO_ROOT / (
            "results/scannet_voxel/scene0000_00/"
            "new_experiments/9361_shutdown_Omni"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=1,
    ),
)


HI_SLAM2_REPLICA_RUN = Run(
    key="replica_hislam2",
    method="HI-SLAM2 (after refinement)",
    dataset_key="replica",
    dataset="Replica office0",
    frame_count=2000,
    directory=REPO_ROOT / "third_party/HI-SLAM2/outputs/replica/office0",
    evaluate_reconstruction=True,
    expected_rendered_depth=0,
    expected_mvs=0,
    expected_omnidata=0,
    representation="gaussian",
    validate_densification_config=False,
    family="HI-SLAM2 monocular",
    native_map_override=(
        REPO_ROOT / "third_party/HI-SLAM2/outputs/replica/office0/3dgs_final.ply"
    ),
    surface_mesh_override=(
        REPO_ROOT
        / "third_party/HI-SLAM2/outputs/replica/office0/tsdf_mesh_after_opt_w2.0.ply"
    ),
    trajectory_override=(
        REPO_ROOT / "third_party/HI-SLAM2/outputs/replica/office0/traj_full.txt"
    ),
)


TANDEM_REPLICA_QUALITY_RUN = Run(
    key="replica_tandem",
    method="TANDEM (SDF)",
    dataset_key="replica",
    dataset="Replica office0",
    frame_count=2000,
    directory=REPO_ROOT / "results/tandem/replica/office0_dataset",
    evaluate_reconstruction=True,
    expected_rendered_depth=0,
    expected_mvs=0,
    expected_omnidata=0,
    representation="mesh",
    validate_densification_config=False,
    family="TANDEM SDF",
    native_map_override=(
        REPO_ROOT / "results/tandem/replica/office0_dataset/mesh.ply"
    ),
    surface_mesh_override=(
        REPO_ROOT / "results/tandem/replica/office0_dataset/mesh.ply"
    ),
    trajectory_override=(
        REPO_ROOT
        / "results/tandem/replica/office0_dataset/CameraTrajectory_TUM.txt"
    ),
    alignment_gt_trajectory_override=(
        REPO_ROOT
        / "results/tandem/replica/office0_dataset/gt_trajectory_matched.txt"
    ),
)


TANDEM_REPLICA_RUNTIME_RUN = Run(
    key="replica_tandem",
    method="TANDEM (SDF)",
    dataset_key="replica",
    dataset="Replica office0",
    frame_count=2000,
    directory=REPO_ROOT / "results/tandem/replica/office0",
    evaluate_reconstruction=False,
    expected_rendered_depth=0,
    expected_mvs=0,
    expected_omnidata=0,
    representation="mesh",
    validate_densification_config=False,
    family="TANDEM SDF",
    native_map_override=(
        REPO_ROOT / "results/tandem/replica/office0/mesh.ply"
    ),
)


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def finite_float(value: Any, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {field} in {path}: {value!r}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {field} in {path}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def discover_one(root: Path, pattern: str, description: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} below {root}; "
            f"found {len(candidates)}"
        )
    require_file(candidates[0], description)
    return candidates[0].resolve()


def saved_config(run: Run) -> Path:
    if not run.validate_densification_config:
        raise RuntimeError(f"{run.key} has no SVRecon densification config")
    return discover_one(run.directory, "*_mono_voxel.yaml", "saved YAML")


def yaml_int(text: str, key: str, default: int = 0) -> int:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*:\s*([01])(?:\s|$)", text
    )
    return default if match is None else int(match.group(1))


def validate_saved_config(run: Run) -> Path:
    path = saved_config(run)
    text = path.read_text(encoding="utf-8")
    actual = (
        yaml_int(text, "Mapper.monocular_rendered_depth_densify"),
        yaml_int(text, "Mapper.monocular_mvs_densify"),
        yaml_int(text, "Mapper.monocular_omnidata_densify"),
    )
    expected = (
        run.expected_rendered_depth,
        run.expected_mvs,
        run.expected_omnidata,
    )
    if actual != expected:
        raise RuntimeError(
            f"Saved YAML does not match {run.method}: {path}; "
            f"expected flags {expected}, found {actual}"
        )
    return path.resolve()


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
        if keyframe_id in values:
            raise RuntimeError(f"Duplicate keyframe {keyframe_id} in {path}")
        values[keyframe_id] = finite_float(tokens[1], "metric", path)
    if not values:
        raise RuntimeError(f"No metrics found in {path}")
    return values


def keyframe_to_frame(run: Run) -> dict[int, int]:
    if run.representation == "gaussian":
        return photoslam_keyframe_to_frame(run)

    path = run.directory / "kf_frame_id_map.txt"
    require_file(path, "keyframe/frame map")
    result: dict[int, int] = {}
    seen_frames: set[int] = set()
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
            keyframe_id, frame_id = (int(token) for token in tokens)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe/frame ID at {path}:{line_number}"
            ) from error
        if not 0 <= frame_id < run.frame_count:
            raise RuntimeError(
                f"Frame {frame_id} is outside [0,{run.frame_count}) in {path}"
            )
        if keyframe_id in result or frame_id in seen_frames:
            raise RuntimeError(f"Duplicate keyframe or frame in {path}")
        result[keyframe_id] = frame_id
        seen_frames.add(frame_id)
    if not result:
        raise RuntimeError(f"No mappings found in {path}")
    return result


def photoslam_keyframe_to_frame(run: Run) -> dict[int, int]:
    path = run.directory / "ply/cameras.json"
    require_file(path, "Photo-SLAM cameras.json")
    try:
        cameras = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse Photo-SLAM cameras: {path}") from error
    if not isinstance(cameras, list):
        raise RuntimeError(f"Expected a camera list in {path}")

    result: dict[int, int] = {}
    seen_frames: set[int] = set()
    for camera in cameras:
        if not isinstance(camera, dict):
            raise RuntimeError(f"Invalid camera entry in {path}")
        try:
            keyframe_id = int(camera["id"])
            image_name = str(camera["img_name"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Camera entry lacks a valid id/img_name in {path}"
            ) from error
        match = re.fullmatch(r"frame(\d+)\.(?:jpg|png)", Path(image_name).name)
        if match is None:
            raise RuntimeError(
                f"Photo-SLAM camera does not reference an exact Replica frame: "
                f"{image_name}"
            )
        frame_id = int(match.group(1))
        if not 0 <= frame_id < run.frame_count:
            raise RuntimeError(
                f"Frame {frame_id} is outside [0,{run.frame_count}) in {path}"
            )
        if keyframe_id in result or frame_id in seen_frames:
            raise RuntimeError(f"Duplicate keyframe or frame in {path}")
        result[keyframe_id] = frame_id
        seen_frames.add(frame_id)
    if not result:
        raise RuntimeError(f"No cameras found in {path}")
    return result


def metric_by_frame(
    values: dict[int, float], mapping: dict[int, int], path: Path
) -> dict[int, float]:
    missing = sorted(set(values).difference(mapping))
    if missing:
        raise RuntimeError(
            f"{path} contains keyframes absent from kf_frame_id_map.txt: "
            f"{missing[:10]}"
        )
    return {mapping[keyframe_id]: value for keyframe_id, value in values.items()}


def photometric_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    for dataset_key in ("tum", "replica", "scannet"):
        runs = [run for run in RUNS if run.dataset_key == dataset_key]
        values_by_run: dict[str, dict[int, dict[str, float]]] = {}
        for run in runs:
            mapping = keyframe_to_frame(run)
            paths = {
                "psnr": run.directory / "psnr.txt",
                "ssim": run.directory / "dssim.txt",
                "render_ms": run.directory / "render_time.txt",
            }
            metrics = {
                name: metric_by_frame(parse_keyed_metric(path), mapping, path)
                for name, path in paths.items()
            }
            frame_ids = set.intersection(*(set(value) for value in metrics.values()))
            if any(set(value) != frame_ids for value in metrics.values()):
                raise RuntimeError(f"Metric frame sets differ for {run.key}")
            values_by_run[run.key] = {
                frame_id: {
                    name: metric[frame_id] for name, metric in metrics.items()
                }
                for frame_id in frame_ids
            }

        common_frames = set.intersection(
            *(set(values_by_run[run.key]) for run in runs)
        )
        if not common_frames:
            raise RuntimeError(f"No exact common frames for {runs[0].dataset}")
        for run in runs:
            rows = [
                values_by_run[run.key][frame_id]
                for frame_id in sorted(common_frames)
            ]
            summaries.append(
                {
                    "dataset": run.dataset,
                    "method": run.method,
                    "evaluated_frames": len(rows),
                    "psnr": statistics.fmean(row["psnr"] for row in rows),
                    "ssim": statistics.fmean(row["ssim"] for row in rows),
                    "render_fps_hz": 1000.0
                    / statistics.fmean(row["render_ms"] for row in rows),
                    "run_dir": str(run.directory),
                }
            )
            for frame_id in sorted(common_frames):
                per_frame.append(
                    {
                        "dataset": run.dataset,
                        "method": run.method,
                        "dataset_frame_id": frame_id,
                        **values_by_run[run.key][frame_id],
                    }
                )
    return summaries, per_frame


def native_map(run: Run) -> Path:
    if run.native_map_override is not None:
        require_file(run.native_map_override, f"native {run.representation} map")
        return run.native_map_override.resolve()
    pattern = (
        "ply/point_cloud/iteration_*/point_cloud.ply"
        if run.representation == "gaussian"
        else "ply/voxel_model/iteration_*/voxel_model.ply"
    )
    return discover_one(
        run.directory,
        pattern,
        f"native {run.representation} map",
    )


def surface_mesh(run: Run) -> Path:
    if run.surface_mesh_override is not None:
        require_file(run.surface_mesh_override, f"{run.representation} surface mesh")
        return run.surface_mesh_override.resolve()
    pattern = (
        "ply/point_cloud/iteration_*/gaussian_surface_mesh.ply"
        if run.representation == "gaussian"
        else "ply/voxel_model/iteration_*/voxel_surface_mesh.ply"
    )
    return discover_one(
        run.directory,
        pattern,
        f"{run.representation} surface mesh",
    )


def hi_precision_mesh(dataset: Dataset, aligned_mesh: Path) -> Path:
    path = (
        aligned_mesh.parent
        / "evaluation_results"
        / f"{dataset.gt_mesh.stem}.precision.ply"
    )
    require_file(path, "HI-SLAM2 ICP-aligned precision mesh")
    return path.resolve()


def save_hi_icp_transform(log_path: Path, output_path: Path) -> Path:
    require_file(log_path, "HI-SLAM2 evaluation log")
    text = log_path.read_text(encoding="utf-8")
    marker = "Rigid Transform Applied to Reconstructed Mesh:"
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise RuntimeError(f"HI-SLAM2 rigid transform is absent from {log_path}")
    matrix_start = text.find("[[", marker_pos)
    matrix_end = text.find("]]", matrix_start)
    if matrix_start < 0 or matrix_end < 0:
        raise RuntimeError(f"Cannot parse HI-SLAM2 rigid transform in {log_path}")

    values = [
        float(token)
        for token in FLOAT_TOKEN_PATTERN.findall(
            text[matrix_start : matrix_end + 2]
        )
    ]
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"Expected 16 finite HI-SLAM2 transform values in {log_path}; "
            f"found {len(values)}"
        )
    if any(abs(values[index]) > 1.0e-8 for index in (12, 13, 14)) or not math.isclose(
        values[15], 1.0, abs_tol=1.0e-8
    ):
        raise RuntimeError(f"Invalid HI-SLAM2 homogeneous transform in {log_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            " ".join(f"{values[4 * row + column]:.17g}" for column in range(4))
            for row in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def evaluate_replica_surface_extras(
    dataset: Dataset,
    experiment: Experiment,
    aligned_mesh: Path,
    gt_trajectory: Path,
) -> dict[str, Any]:
    precision_mesh = hi_precision_mesh(dataset, aligned_mesh)
    output_dir = experiment.directory / "complete_surface_metrics"
    command = [
        str(MESH_EVAL),
        "--eval_mode=current",
        f"--recon={precision_mesh}",
        f"--gt={dataset.gt_mesh}",
        f"--out={output_dir}",
        f"--tau_cm={100.0 * DISTANCE_THRESHOLD_M:g}",
        "--eval_depth_mesh=1",
        f"--traj={gt_trajectory}",
        "--traj_mode=c2w",
        f"--img_w={REPLICA_DEPTH_INTRINSICS['img_w']}",
        f"--img_h={REPLICA_DEPTH_INTRINSICS['img_h']}",
        f"--fx={REPLICA_DEPTH_INTRINSICS['fx']}",
        f"--fy={REPLICA_DEPTH_INTRINSICS['fy']}",
        f"--cx={REPLICA_DEPTH_INTRINSICS['cx']}",
        f"--cy={REPLICA_DEPTH_INTRINSICS['cy']}",
        "--frame_stride=1",
        f"--max_frames={REPLICA_DEPTH_MAX_FRAMES}",
        "--near=0.05",
        "--far=20.0",
        "--seed=0",
        f"--recon_samples={REPLICA_SURFACE_SAMPLES}",
        f"--gt_samples={REPLICA_SURFACE_SAMPLES}",
        "--eval_floaters=1",
        f"--floater_samples={REPLICA_FLOATER_SAMPLES}",
        "--floater_bin_cm=1.0",
        "--floater_max_cm=50.0",
        "--eval_gaussian_support=0",
        "--eval_voxel_support=0",
    ]
    run_logged(command, REPO_ROOT, output_dir / "mesh_eval.log")
    metrics_path = output_dir / "mesh_eval.json"
    metrics = load_json(metrics_path)
    if not metrics.get("depth_from_mesh_success"):
        raise RuntimeError(f"Depth-from-mesh evaluation failed: {metrics_path}")
    return {
        "depth_l1_m": finite_float(metrics.get("depth_l1_m"), "depth_l1_m", metrics_path),
        "depth_frames": int(metrics.get("depth_frames_used", 0)),
        "mesh_floater_ratio": finite_float(
            metrics.get("surface_floater_ratio"),
            "surface_floater_ratio",
            metrics_path,
        ),
        "hi_precision_mesh": str(precision_mesh),
        "surface_metrics_file": str(metrics_path.resolve()),
    }


def evaluate_replica_primitive_floaters(
    run: Run,
    dataset: Dataset,
    experiment: Experiment,
    recon_trajectory: Path,
    gt_trajectory: Path,
    hi_icp_transform: Path,
) -> dict[str, Any]:
    map_path = native_map(run)
    output_dir = experiment.directory / "complete_primitive_metrics"
    gaussian_support = run.representation == "gaussian"
    metric_prefix = "gaussian_support" if gaussian_support else "voxel_support"
    command = [
        str(MESH_EVAL),
        "--eval_mode=current",
        f"--recon={map_path}",
        f"--gt={dataset.gt_mesh}",
        f"--out={output_dir}",
        f"--tau_cm={100.0 * DISTANCE_THRESHOLD_M:g}",
        "--align_recon_to_gt=1",
        f"--traj={gt_trajectory}",
        "--traj_mode=c2w",
        f"--recon_traj_tum={recon_trajectory}",
        f"--post_align_transform={hi_icp_transform}",
        "--recon_samples=1",
        f"--gt_samples={REPLICA_SURFACE_SAMPLES}",
        "--eval_floaters=0",
        f"--eval_gaussian_support={int(gaussian_support)}",
        f"--eval_voxel_support={int(not gaussian_support)}",
        "--gaussian_support_sigma=3.0",
        f"--support_samples_per_primitive={REPLICA_SUPPORT_SAMPLES_PER_PRIMITIVE}",
        "--floater_bin_cm=1.0",
        "--floater_max_cm=50.0",
        "--seed=0",
    ]
    run_logged(command, REPO_ROOT, output_dir / "mesh_eval.log")
    metrics_path = output_dir / "mesh_eval.json"
    metrics = load_json(metrics_path)
    if not metrics.get(f"{metric_prefix}_evaluated"):
        raise RuntimeError(f"Primitive floater evaluation failed: {metrics_path}")
    return {
        "primitive_floater_ratio": finite_float(
            metrics.get(f"{metric_prefix}_floater_ratio"),
            f"{metric_prefix}_floater_ratio",
            metrics_path,
        ),
        "primitive_count": int(
            metrics.get(f"{metric_prefix}_primitive_count", 0)
        ),
        "primitive_support": (
            "3-sigma Gaussian ellipsoid"
            if gaussian_support else "SVRecon zero-crossing cell"
        ),
        "native_map": str(map_path),
        "primitive_metrics_file": str(metrics_path.resolve()),
    }


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as ply_file:
        for raw_line in ply_file:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise RuntimeError(f"PLY has no element vertex declaration: {path}")


def shutdown_iteration(run: Run) -> int:
    match = re.match(r"^(\d+)_shutdown", run.directory.name)
    if match is None:
        raise RuntimeError(f"Cannot parse shutdown iteration: {run.directory}")
    return int(match.group(1))


def run_tracking_fps(run: Run, runtime: dict[str, Any] | None) -> float | None:
    if runtime is not None and runtime.get("tracking_fps_hz") is not None:
        return finite_float(
            runtime["tracking_fps_hz"], "tracking_fps_hz",
            run.directory / "runtime_metrics.json",
        )

    candidates = [run.directory / "TrackingTime.txt"]
    candidates.extend(
        parent / "TrackingTime.txt"
        for parent in list(run.directory.parents)[:3]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    times = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = finite_float(stripped, f"tracking time line {line_number}", path)
        if value > 0.0:
            times.append(value)
    if len(times) != run.frame_count:
        raise RuntimeError(
            f"Expected {run.frame_count} tracking times in {path}; "
            f"found {len(times)}"
        )
    return len(times) / sum(times)


def run_render_fps(run: Run) -> float | None:
    path = run.directory / "render_time.txt"
    if not path.is_file():
        return None
    render_times_ms = list(parse_keyed_metric(path).values())
    mean_ms = statistics.fmean(render_times_ms)
    if mean_ms <= 0.0:
        raise RuntimeError(f"Non-positive mean render time in {path}")
    return 1000.0 / mean_ms


def run_gpu_peak_reserved_mb(
    run: Run, runtime: dict[str, Any] | None
) -> float | None:
    path = run.directory / "GpuPeakUsageMB.txt"
    if path.is_file():
        match = re.search(
            r"(?m)^Peak reserved \(MB\):\s*(\S+)\s*$",
            path.read_text(encoding="utf-8"),
        )
        if match is None:
            raise RuntimeError(f"Missing peak reserved memory in {path}")
        return finite_float(match.group(1), "peak reserved MB", path)
    if runtime is None or runtime.get("gpu_memory_reserved_mb") is None:
        return None
    return finite_float(
        runtime["gpu_memory_reserved_mb"],
        "gpu_memory_reserved_mb",
        run.directory / "runtime_metrics.json",
    )


def computational_result(run: Run, config: Path | None) -> dict[str, Any]:
    map_path = native_map(run)
    primitive_count = ply_vertex_count(map_path)
    runtime_path = run.directory / "runtime_metrics.json"
    runtime: dict[str, Any] | None = None
    if runtime_path.is_file():
        runtime = load_json(runtime_path)
        frames = int(runtime.get("frames", 0))
        if frames != run.frame_count:
            raise RuntimeError(
                f"Expected {run.frame_count} frames in {runtime_path}; "
                f"found {frames}"
            )
    iterations: int | None = None
    if runtime is not None and runtime.get("iterations") is not None:
        iterations = int(runtime["iterations"])
    elif run.representation != "mesh":
        iterations = shutdown_iteration(run)
    keyframes = (
        int(runtime.get("keyframes", 0))
        if runtime is not None and runtime.get("keyframes") is not None
        else len(keyframe_to_frame(run))
    )
    return {
        "run_key": run.key,
        "dataset": run.dataset,
        "method": run.method,
        "frames": run.frame_count,
        "keyframes": keyframes,
        "iterations": iterations,
        "primitive_type": run.representation,
        "primitive_count": primitive_count,
        "runtime_seconds": (
            finite_float(runtime["total_seconds"], "total_seconds", runtime_path)
            if runtime is not None else None
        ),
        "system_fps_hz": (
            run.frame_count
            / finite_float(runtime["total_seconds"], "total_seconds", runtime_path)
            if runtime is not None else None
        ),
        "tracking_fps_hz": run_tracking_fps(run, runtime),
        "render_fps_hz": run_render_fps(run),
        "map_size_mb": map_path.stat().st_size / (1024.0 * 1024.0),
        "gpu_memory_allocated_mb": (
            finite_float(
                runtime["gpu_memory_allocated_mb"],
                "gpu_memory_allocated_mb",
                runtime_path,
            )
            if runtime is not None else None
        ),
        "gpu_memory_reserved_mb": run_gpu_peak_reserved_mb(run, runtime),
        "gpu_process_peak_used_mb": (
            finite_float(
                runtime["gpu_process_peak_used_mb"],
                "gpu_process_peak_used_mb",
                runtime_path,
            )
            if runtime is not None
            and runtime.get("gpu_process_peak_used_mb") is not None
            else None
        ),
        "runtime_available": runtime is not None,
        "saved_config": str(config) if config is not None else None,
        "native_map": str(map_path),
    }


def hislam2_computational_result() -> dict[str, Any]:
    run = HI_SLAM2_REPLICA_RUN
    map_path = native_map(run)
    runtime_path = run.directory / "runtime_metrics.json"
    runtime = load_json(runtime_path)
    after_opt = runtime.get("after_opt")
    if not isinstance(after_opt, dict):
        raise RuntimeError(f"Missing after_opt object in {runtime_path}")
    frames = int(runtime.get("frames", 0))
    if frames != run.frame_count:
        raise RuntimeError(
            f"Expected {run.frame_count} frames in {runtime_path}; found {frames}"
        )
    total_seconds = finite_float(
        runtime.get("total_seconds"), "total_seconds", runtime_path
    )
    return {
        "run_key": run.key,
        "dataset": run.dataset,
        "method": run.method,
        "frames": frames,
        "keyframes": int(runtime.get("keyframes", 0)),
        "iterations": None,
        "primitive_type": run.representation,
        "primitive_count": ply_vertex_count(map_path),
        "runtime_seconds": total_seconds,
        "system_fps_hz": frames / total_seconds,
        "tracking_fps_hz": finite_float(
            runtime.get("tracking_fps_hz"), "tracking_fps_hz", runtime_path
        ),
        "render_fps_hz": None,
        "map_size_mb": map_path.stat().st_size / (1024.0 * 1024.0),
        "gpu_memory_allocated_mb": finite_float(
            after_opt.get("gpu_memory_allocated_mb"),
            "after_opt.gpu_memory_allocated_mb",
            runtime_path,
        ),
        "gpu_memory_reserved_mb": finite_float(
            after_opt.get("gpu_memory_reserved_mb"),
            "after_opt.gpu_memory_reserved_mb",
            runtime_path,
        ),
        "gpu_process_peak_used_mb": None,
        "runtime_available": True,
        "saved_config": None,
        "native_map": str(map_path),
    }


def reconstruction_experiment(run: Run, dataset_tag: str) -> Experiment:
    trajectory = (
        run.trajectory_override
        if run.trajectory_override is not None
        else run.directory / "CameraTrajectory_TUM.txt"
    )
    require_file(trajectory, "run-local reconstruction trajectory")
    if run.alignment_gt_trajectory_override is not None:
        require_file(
            run.alignment_gt_trajectory_override,
            "frame-matched alignment GT trajectory",
        )
    return Experiment(
        name=run.method,
        family=run.family,
        directory=(
            OUTPUT_DIR / "reconstruction" / run.dataset_key
            / dataset_tag / run.key
        ),
        mesh=surface_mesh(run),
        recon_trajectory=trajectory.resolve(),
        gt_trajectory=(
            run.alignment_gt_trajectory_override.resolve()
            if run.alignment_gt_trajectory_override is not None
            else None
        ),
    )


def reconstruction_datasets() -> tuple[Dataset, ...]:
    replica_runs = [run for run in RUNS if run.dataset_key == "replica"]
    replica_runs.append(HI_SLAM2_REPLICA_RUN)
    replica_runs.append(TANDEM_REPLICA_QUALITY_RUN)
    scannet_runs = [run for run in RUNS if run.dataset_key == "scannet"]
    replica_gt_trajectory = REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
    replica_placeholder = replica_runs[0].directory / "CameraTrajectory_TUM.txt"
    scannet_placeholder = scannet_runs[0].directory / "CameraTrajectory_TUM.txt"
    return (
        Dataset(
            name="Replica office0 (culled GT)",
            gt_mesh=(
                REPO_ROOT
                / "third_party/ESLAM/cull_replica_mesh/office0_culled.ply"
            ),
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=replica_placeholder,
            artifact_tag="culled",
            summary_dir=OUTPUT_DIR / "reconstruction/replica_culled",
            experiments=tuple(
                reconstruction_experiment(run, "culled")
                for run in replica_runs
            ),
        ),
        Dataset(
            name="Replica office0 (original GT)",
            gt_mesh=REPO_ROOT / "scripts/data/Replica/office0_mesh.ply",
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=replica_placeholder,
            artifact_tag="original",
            summary_dir=OUTPUT_DIR / "reconstruction/replica_original",
            experiments=tuple(
                reconstruction_experiment(run, "original")
                for run in replica_runs
            ),
        ),
        Dataset(
            name="ScanNet scene0000_00",
            gt_mesh=(
                REPO_ROOT / "scripts/data/ScanNet/scans/scene0000_00"
                / "scene0000_00_vh_clean_2.ply"
            ),
            gt_pose_directory=(
                REPO_ROOT / "scripts/data/ScanNet/scans/scene0000_00/pose"
            ),
            shared_recon_trajectory=scannet_placeholder,
            summary_dir=OUTPUT_DIR / "reconstruction/scannet",
            experiments=tuple(
                reconstruction_experiment(run, "scene0000_00")
                for run in scannet_runs
            ),
        ),
    )


def evaluate_reconstruction() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    replica_complete: list[dict[str, Any]] = []
    completed_replica_keys: set[str] = set()
    runs_by_key = {
        run.key: run
        for run in (*RUNS, HI_SLAM2_REPLICA_RUN, TANDEM_REPLICA_QUALITY_RUN)
    }
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    for dataset in reconstruction_datasets():
        print(f"\n=== Reconstruction: {dataset.name} ===")
        require_file(dataset.gt_mesh, f"GT mesh for {dataset.name}")
        gt_trajectory = materialize_gt_trajectory(dataset)
        dataset_results: list[dict[str, Any]] = []
        for experiment in dataset.experiments:
            assert experiment.mesh is not None
            assert experiment.recon_trajectory is not None
            experiment.directory.mkdir(parents=True, exist_ok=True)
            aligned_mesh = align_mesh(
                dataset,
                experiment,
                experiment.mesh,
                experiment.recon_trajectory,
                (
                    experiment.gt_trajectory
                    if experiment.gt_trajectory is not None
                    else gt_trajectory
                ),
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

            run = runs_by_key.get(experiment.directory.name)
            if (
                dataset.artifact_tag == "culled"
                and run is not None
                and run.key in REPLICA_COMPLETE_RUN_LABELS
            ):
                hi_icp_transform = save_hi_icp_transform(
                    result_path.with_suffix(".log"),
                    experiment.directory / "hi_icp_transform.txt",
                )
                surface_extras = evaluate_replica_surface_extras(
                    dataset,
                    experiment,
                    aligned_mesh,
                    gt_trajectory,
                )
                primitive_extras = (
                    None
                    if run.representation == "mesh"
                    else evaluate_replica_primitive_floaters(
                        run,
                        dataset,
                        experiment,
                        experiment.recon_trajectory,
                        gt_trajectory,
                        hi_icp_transform,
                    )
                )
                depth_l1_m = surface_extras["depth_l1_m"]
                mesh_floater_ratio = surface_extras["mesh_floater_ratio"]
                primitive_floater_ratio = (
                    primitive_extras["primitive_floater_ratio"]
                    if primitive_extras is not None else None
                )
                replica_complete.append(
                    {
                        "experiment": REPLICA_COMPLETE_RUN_LABELS[run.key],
                        "method": run.method,
                        "accuracy_m": result["accuracy_m"],
                        "completeness_m": result["completeness_m"],
                        "depth_l1_m": depth_l1_m,
                        "depth_l1_cm": 100.0 * depth_l1_m,
                        f"precision_{suffix}": result[f"precision_{suffix}"],
                        f"recall_{suffix}": result[f"recall_{suffix}"],
                        f"completion_{suffix}_percent": result[
                            f"completion_{suffix}_percent"
                        ],
                        f"fscore_{suffix}": result[f"fscore_{suffix}"],
                        f"mesh_floaters_{suffix}_ratio": mesh_floater_ratio,
                        f"mesh_floaters_{suffix}_percent": (
                            100.0 * mesh_floater_ratio
                        ),
                        f"primitive_floaters_{suffix}_ratio": (
                            primitive_floater_ratio
                        ),
                        f"primitive_floaters_{suffix}_percent": (
                            100.0 * primitive_floater_ratio
                            if primitive_floater_ratio is not None else None
                        ),
                        "depth_frames": surface_extras["depth_frames"],
                        "primitive_count": (
                            primitive_extras["primitive_count"]
                            if primitive_extras is not None else None
                        ),
                        "primitive_support": (
                            primitive_extras["primitive_support"]
                            if primitive_extras is not None
                            else "not serialized by TANDEM"
                        ),
                        "raw_surface_mesh": str(experiment.mesh),
                        "native_map": (
                            primitive_extras["native_map"]
                            if primitive_extras is not None
                            else str(native_map(run))
                        ),
                        "trajectory": str(experiment.recon_trajectory),
                        "hi_metrics_file": str(result_path),
                        "hi_precision_mesh": surface_extras[
                            "hi_precision_mesh"
                        ],
                        "surface_metrics_file": surface_extras[
                            "surface_metrics_file"
                        ],
                        "primitive_metrics_file": (
                            primitive_extras["primitive_metrics_file"]
                            if primitive_extras is not None else None
                        ),
                        "hi_icp_transform": str(hi_icp_transform),
                    }
                )
                completed_replica_keys.add(run.key)
        write_summaries(
            dataset,
            gt_trajectory,
            dataset_results,
            DISTANCE_THRESHOLD_M,
        )
    expected_keys = set(REPLICA_COMPLETE_RUN_LABELS)
    if completed_replica_keys != expected_keys:
        raise RuntimeError(
            "Incomplete Replica table: expected "
            f"{sorted(expected_keys)}, found {sorted(completed_replica_keys)}"
        )
    return results, replica_complete


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


def optional_number(value: Any, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def photometric_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Method | Common frames | PSNR ↑ | SSIM ↑ | Render FPS ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['evaluated_frames']} | {row['psnr']:.4f} | "
            f"{row['ssim']:.4f} | {row['render_fps_hz']:.2f} |"
        )
    return "\n".join(lines)


def computational_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Method | Iterations | Primitives | System FPS ↑ | "
        "Tracking FPS ↑ | Render FPS ↑ | Map MB ↓ | GPU reserved MB ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{optional_number(row['iterations'], 0)} | "
            f"{row['primitive_count']} | "
            f"{optional_number(row['system_fps_hz'])} | "
            f"{optional_number(row['tracking_fps_hz'])} | "
            f"{optional_number(row['render_fps_hz'])} | "
            f"{row['map_size_mb']:.2f} | "
            f"{optional_number(row['gpu_memory_reserved_mb'])} |"
        )
    return "\n".join(lines)


def reconstruction_table(rows: list[dict[str, Any]]) -> str:
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    lines = [
        "| Dataset | Method | Acc. m ↓ | Compl. m ↓ | Chamfer m ↓ | "
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


def replica_complete_table(rows: list[dict[str, Any]]) -> str:
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    lines = [
        "| Experiment | Acc. m ↓ | Compl. m ↓ | Depth L1 cm ↓ | "
        f"Prec@{suffix} ↑ | Recall@{suffix} ↑ | Completion@{suffix} ↑ | "
        f"F@{suffix} ↑ | Mesh floaters@{suffix} ↓ | "
        f"Primitive floaters@{suffix} ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['accuracy_m']:.6f} | "
            f"{row['completeness_m']:.6f} | {row['depth_l1_cm']:.4f} | "
            f"{row[f'precision_{suffix}']:.4f} | "
            f"{row[f'recall_{suffix}']:.4f} | "
            f"{row[f'completion_{suffix}_percent']:.2f}% | "
            f"{row[f'fscore_{suffix}']:.4f} | "
            f"{row[f'mesh_floaters_{suffix}_percent']:.2f}% | "
            f"{optional_number(row[f'primitive_floaters_{suffix}_percent'])}"
            f"{'%' if row[f'primitive_floaters_{suffix}_percent'] is not None else ''} |"
        )
    return "\n".join(lines)


def replica_table1_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    by_label = {row["experiment"]: row for row in rows}
    result = []
    for run_key in REPLICA_PRESENTATION_RUN_ORDER:
        label = REPLICA_COMPLETE_RUN_LABELS[run_key]
        if label not in by_label:
            raise RuntimeError(f"Replica Table 1 is missing {label}")
        row = by_label[label]
        result.append(
            {
                "experiment": label,
                "accuracy_m": row["accuracy_m"],
                "completeness_m": row["completeness_m"],
                "depth_l1_cm": row["depth_l1_cm"],
                f"precision_{suffix}": row[f"precision_{suffix}"],
                f"recall_{suffix}": row[f"recall_{suffix}"],
                f"completion_{suffix}_percent": row[
                    f"completion_{suffix}_percent"
                ],
                f"fscore_{suffix}": row[f"fscore_{suffix}"],
                f"floaters_{suffix}_percent": row[
                    f"mesh_floaters_{suffix}_percent"
                ],
            }
        )
    return result


def replica_table1_markdown(rows: list[dict[str, Any]]) -> str:
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    lines = [
        "| Experiment | Acc ↓ | Compl. ↓ | Depth L1 (cm) ↓ | "
        f"Prec@{suffix} ↑ | Recall@{suffix} ↑ | Completion@{suffix} ↑ | "
        f"F@{suffix} ↑ | Floaters ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['accuracy_m']:.4f} | "
            f"{row['completeness_m']:.4f} | {row['depth_l1_cm']:.4f} | "
            f"{row[f'precision_{suffix}']:.4f} | "
            f"{row[f'recall_{suffix}']:.4f} | "
            f"{row[f'completion_{suffix}_percent']:.2f}% | "
            f"{row[f'fscore_{suffix}']:.4f} | "
            f"{row[f'floaters_{suffix}_percent']:.2f}% |"
        )
    return "\n".join(lines)


def replica_table2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replica_rows = {
        row["method"]: row for row in rows if row["dataset"] == "Replica office0"
    }
    method_labels = (
        ("Rendered-depth densification", "Ours"),
        ("TANDEM MVS", "Ours + MVS"),
        ("Photo-SLAM", "Photo-SLAM (3D GS)"),
    )
    result = []
    for method, label in method_labels:
        if method not in replica_rows:
            raise RuntimeError(f"Replica Table 2 is missing {method}")
        row = replica_rows[method]
        result.append(
            {
                "experiment": label,
                "psnr": row["psnr"],
                "ssim": row["ssim"],
            }
        )

    require_file(HI_SLAM2_REPLICA_PHOTOMETRIC, "HI-SLAM2 photometric result")
    payload = json.loads(HI_SLAM2_REPLICA_PHOTOMETRIC.read_text(encoding="utf-8"))
    psnr = float(payload["mean_psnr"])
    ssim = float(payload["mean_ssim"])
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise RuntimeError(
            f"Non-finite HI-SLAM2 photometric result in "
            f"{HI_SLAM2_REPLICA_PHOTOMETRIC}"
        )
    result.append(
        {
            "experiment": "HI-SLAM2 (3D GS)",
            "psnr": psnr,
            "ssim": ssim,
        }
    )
    return result


def replica_table2_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Experiment | PSNR ↑ | SSIM ↑ |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['psnr']:.4f} | {row['ssim']:.4f} |"
        )
    return "\n".join(lines)


def replica_table3_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {row["run_key"]: row for row in rows}
    result = []
    for run_key in REPLICA_PRESENTATION_RUN_ORDER:
        if run_key not in by_key:
            raise RuntimeError(f"Replica Table 3 is missing {run_key}")
        row = by_key[run_key]
        result.append(
            {
                "experiment": REPLICA_TABLE3_LABELS[run_key],
                "system_fps_hz": row["system_fps_hz"],
                "tracking_fps_hz": row["tracking_fps_hz"],
                "render_fps_hz": row["render_fps_hz"],
                "map_size_mb": row["map_size_mb"],
                "gpu_memory_reserved_mb": row["gpu_memory_reserved_mb"],
            }
        )
    return result


def replica_table3_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Experiment | System FPS | Tracking FPS | Render FPS | Map MB | "
        "GPU reserved MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {optional_number(row['system_fps_hz'])} | "
            f"{optional_number(row['tracking_fps_hz'])} | "
            f"{optional_number(row['render_fps_hz'])} | "
            f"{optional_number(row['map_size_mb'])} | "
            f"{optional_number(row['gpu_memory_reserved_mb'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    require_file(MESH_EVAL, "mesh_eval executable")
    require_file(HI_SLAM2_EVALUATOR, "HI-SLAM2 evaluator")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python")
    configs: dict[str, Path | None] = {}
    for run in RUNS:
        require_directory(run.directory, "ablation run directory")
        configs[run.key] = (
            validate_saved_config(run)
            if run.validate_densification_config else None
        )
    require_directory(
        TANDEM_REPLICA_QUALITY_RUN.directory,
        "TANDEM Replica dataset-preset run directory",
    )
    require_directory(
        TANDEM_REPLICA_RUNTIME_RUN.directory,
        "TANDEM Replica runtime-preset run directory",
    )

    photometric, photometric_per_frame = photometric_results()
    computational = [
        computational_result(run, configs[run.key]) for run in RUNS
    ]
    computational.append(hislam2_computational_result())
    computational.append(computational_result(TANDEM_REPLICA_RUNTIME_RUN, None))
    reconstruction, replica_complete = evaluate_reconstruction()

    photo_md = photometric_table(photometric)
    compute_md = computational_table(computational)
    reconstruction_md = reconstruction_table(reconstruction)
    replica_complete_md = replica_complete_table(replica_complete)
    table1 = replica_table1_rows(replica_complete)
    table1_md = replica_table1_markdown(table1)
    table2 = replica_table2_rows(photometric)
    table2_md = replica_table2_markdown(table2)
    table3 = replica_table3_rows(computational)
    table3_md = replica_table3_markdown(table3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT_DIR / "photometric_metrics.csv", photometric)
    write_csv(
        OUTPUT_DIR / "photometric_metrics_per_frame.csv",
        photometric_per_frame,
    )
    write_json(
        OUTPUT_DIR / "photometric_metrics.json",
        {
            "aggregation": "arithmetic mean over exact common dataset frames",
            "ssim_source": "dssim.txt stores SSIM despite its legacy filename",
        },
        photometric,
    )
    (OUTPUT_DIR / "photometric_metrics.md").write_text(
        photo_md + "\n", encoding="utf-8"
    )

    write_csv(OUTPUT_DIR / "computational_metrics.csv", computational)
    write_json(
        OUTPUT_DIR / "computational_metrics.json",
        {
            "system_fps": "dataset frames / recorded mapping total_seconds",
            "tracking_fps": (
                "Photo-SLAM/SVRecon use ORB TrackingTime.txt; HI-SLAM2 uses "
                "its recorded tracker time; TANDEM uses its timed DSO "
                "trackNewCoarse frontend"
            ),
            "render_fps": (
                "mean standalone learned-map render timing when available; "
                "N/A for TANDEM and HI-SLAM2 because no comparable saved timing exists"
            ),
            "map_size": (
                "actual run-local native map PLY size; TANDEM uses the binary "
                "PLY conversion of its only serialized map artifact, mesh.obj"
            ),
            "gpu_memory": "peak recorded PyTorch CUDA caching-allocator reserved memory",
            "tandem_process_gpu_memory": (
                "gpu_process_peak_used_mb additionally records nvidia-smi "
                "per-process memory, including TANDEM custom CUDA TSDF buffers"
            ),
            "missing_runtime": (
                "TUM rendered-depth run has no runtime_metrics.json; "
                "runtime and GPU values are null"
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
            "completion_rate": "identical to recall and reported as percent",
            "tum": "omitted because no GT reconstruction mesh is provided",
        },
        reconstruction,
    )
    (OUTPUT_DIR / "reconstruction_metrics.md").write_text(
        reconstruction_md + "\n", encoding="utf-8"
    )

    write_csv(OUTPUT_DIR / "replica_complete_metrics.csv", replica_complete)
    write_json(
        OUTPUT_DIR / "replica_complete_metrics.json",
        {
            "ground_truth": (
                "Replica office0 culled mesh from ESLAM, evaluated through "
                "the HI-SLAM2 reconstruction protocol"
            ),
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "alignment": (
                "trajectory Sim(3), followed by the exact rigid ICP transform "
                "computed and applied by HI-SLAM2"
            ),
            "geometry_metrics": "HI-SLAM2 scripts/eval_recon.py --eval_3d",
            "depth_l1": (
                "mean absolute mesh-rendered depth error over common valid "
                "pixels in 1000 deterministic Replica trajectory views"
            ),
            "mesh_floaters": (
                "fraction of reconstructed surface samples farther than 5 cm "
                "from GT after the final HI-SLAM2 alignment"
            ),
            "primitive_floaters": (
                "fraction of representation-native primitive supports whose "
                "maximum boundary distance exceeds 5 cm after the same final "
                "HI-SLAM2 alignment: zero-crossing SVRecon cell cubes or "
                "Photo-SLAM 3-sigma Gaussian ellipsoids; N/A for TANDEM "
                "because it does not serialize the native TSDF cells"
            ),
            "completion_rate": "identical to recall and reported as percent",
            "surface_samples": REPLICA_SURFACE_SAMPLES,
            "floater_samples": REPLICA_FLOATER_SAMPLES,
            "support_samples_per_primitive": (
                REPLICA_SUPPORT_SAMPLES_PER_PRIMITIVE
            ),
            "depth_frames": REPLICA_DEPTH_MAX_FRAMES,
            "random_seed": 0,
        },
        replica_complete,
    )
    (OUTPUT_DIR / "replica_complete_metrics.md").write_text(
        replica_complete_md + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "replica_table1.csv", table1)
    (OUTPUT_DIR / "replica_table1.md").write_text(
        table1_md + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "replica_table2.csv", table2)
    (OUTPUT_DIR / "replica_table2.md").write_text(
        table2_md + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "replica_table3.csv", table3)
    (OUTPUT_DIR / "replica_table3.md").write_text(
        table3_md + "\n", encoding="utf-8"
    )

    summary = (
        "# Monocular Densification Ablation\n\n"
        "## Photometric Quality\n\n"
        f"{photo_md}\n\n"
        "## Computational Performance\n\n"
        f"{compute_md}\n\n"
        "## Reconstruction\n\n"
        f"{reconstruction_md}\n\n"
        "## Replica Complete Reconstruction Table (Culled GT)\n\n"
        f"{replica_complete_md}\n\n"
        "## Replica Presentation Table 1\n\n"
        f"{table1_md}\n\n"
        "## Replica Presentation Table 2\n\n"
        f"{table2_md}\n\n"
        "## Replica Presentation Table 3\n\n"
        f"{table3_md}\n"
    )
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\nSaved ablation evaluation to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
