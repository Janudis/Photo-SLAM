#!/usr/bin/env python3
"""Evaluate paired monocular SVRecon and Photo-SLAM runs.

Expected layout below --runs-root:

  tum/fr1_desk/{ours,photoslam}
  replica/office0/{ours,photoslam}
  scannet/scene0000_00/{ours,photoslam}

Each method directory must contain exactly one numeric *_shutdown directory.
The script always recomputes its outputs and stops on the first invalid input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_reconstructions_hislam2 import (  # noqa: E402
    Dataset as ReconstructionDataset,
    Experiment as ReconstructionExperiment,
    align_mesh,
    evaluate_aligned_mesh,
    materialize_gt_trajectory,
    require_file,
    result_from_metrics,
    threshold_label,
    write_summaries,
)


DEFAULT_RUNS_ROOT = (
    REPO_ROOT / "results/monocular_benchmark/run_01"
)
MONOGS_PYTHON = (
    Path.home() / "miniconda3/envs/MonoGS/bin/python"
)
MONOGS_EXPORTER = (
    REPO_ROOT / "evaluation/export_monogs_reconstruction.py"
)
DISTANCE_THRESHOLD_M = 0.05
SHUTDOWN_PATTERN = re.compile(r"^(\d+)_shutdown$")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    relative_run_dir: Path
    sequence_dir: Path
    image_list: Path | None


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    family: str
    representation: str


@dataclass(frozen=True)
class RunArtifacts:
    dataset: DatasetSpec
    method: MethodSpec
    run_dir: Path
    shutdown_dir: Path
    runtime_metrics: Path
    trajectory: Path
    native_map: Path
    surface_mesh: Path | None


DATASETS = (
    DatasetSpec(
        key="tum",
        name="TUM fr1/desk",
        relative_run_dir=Path("tum/fr1_desk"),
        sequence_dir=(
            REPO_ROOT / "scripts/data/rgbd_dataset_freiburg1_desk"
        ),
        image_list=(
            REPO_ROOT
            / "scripts/data/rgbd_dataset_freiburg1_desk/rgb.txt"
        ),
    ),
    DatasetSpec(
        key="replica",
        name="Replica office0",
        relative_run_dir=Path("replica/office0"),
        sequence_dir=REPO_ROOT / "scripts/data/Replica/office0",
        image_list=None,
    ),
    DatasetSpec(
        key="scannet",
        name="ScanNet scene0000_00",
        relative_run_dir=Path("scannet/scene0000_00"),
        sequence_dir=(
            REPO_ROOT
            / "scripts/data/ScanNet/scans/scene0000_00"
        ),
        image_list=(
            REPO_ROOT
            / "scripts/data/ScanNet/scans/scene0000_00/association.txt"
        ),
    ),
)

METHODS = (
    MethodSpec(
        key="ours",
        name="Ours",
        family="SVRecon",
        representation="voxels",
    ),
    MethodSpec(
        key="photoslam",
        name="Photo-SLAM",
        family="Photo-SLAM",
        representation="gaussians",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate computational, photometric, and reconstruction "
            "metrics for paired monocular runs."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"benchmark run root (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--monogs-replica-run",
        type=Path,
        help=(
            "completed timestamped MonoGS Replica run; when provided, "
            "its after-refinement map is re-exported and added only to "
            "the Replica reconstruction tables"
        ),
    )
    return parser.parse_args()


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def discover_shutdown(run_dir: Path) -> Path:
    require_directory(run_dir, "method run directory")
    candidates = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and SHUTDOWN_PATTERN.fullmatch(path.name)
    ]
    if len(candidates) != 1:
        rendered = ", ".join(sorted(path.name for path in candidates))
        raise RuntimeError(
            f"Expected exactly one numeric *_shutdown directory in "
            f"{run_dir}; found {len(candidates)}"
            + (f": {rendered}" if rendered else "")
        )
    return candidates[0].resolve()


def discover_one(root: Path, pattern: str, description: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} below {root}; "
            f"found {len(candidates)}"
        )
    require_file(candidates[0], description)
    return candidates[0].resolve()


def load_json(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return value


def prepare_monogs_replica_experiment(
    run_dir: Path,
) -> ReconstructionExperiment:
    run_dir = run_dir.expanduser().resolve()
    require_directory(run_dir, "MonoGS Replica run directory")
    require_file(MONOGS_PYTHON, "MonoGS Python interpreter")
    require_file(MONOGS_EXPORTER, "MonoGS reconstruction exporter")

    command = [
        str(MONOGS_PYTHON),
        str(MONOGS_EXPORTER),
        "--run-dir",
        str(run_dir),
    ]
    print("\n=== Exporting MonoGS Replica reconstruction ===")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    export_dir = run_dir / "reconstruction_eval"
    metadata = load_json(
        export_dir / "export_metadata.json",
        "MonoGS reconstruction export metadata",
    )
    resolution = metadata.get("input_resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(isinstance(value, int) for value in resolution)
    ):
        raise RuntimeError(
            "MonoGS export metadata has no valid input resolution"
        )

    mesh = export_dir / "monogs_surface_mesh.ply"
    recon_trajectory = export_dir / "CameraTrajectory_TUM.txt"
    gt_trajectory = export_dir / "gt_trajectory_row_major.txt"
    require_file(mesh, "MonoGS rendered-depth TSDF mesh")
    require_file(recon_trajectory, "MonoGS keyframe trajectory")
    require_file(gt_trajectory, "MonoGS matched GT keyframe trajectory")
    return ReconstructionExperiment(
        name=(
            f"MonoGS {resolution[0]}x{resolution[1]} "
            "(after refinement)"
        ),
        family="3D-GS",
        directory=export_dir,
        mesh=mesh,
        recon_trajectory=recon_trajectory,
        gt_trajectory=gt_trajectory,
    )


def resolve_native_map(
    runtime: dict[str, Any],
    shutdown_dir: Path,
    method: MethodSpec,
) -> Path:
    map_path_value = runtime.get("map_path")
    if isinstance(map_path_value, str) and map_path_value:
        map_path = Path(map_path_value)
        candidates = [map_path]
        if not map_path.is_absolute():
            candidates.extend(
                (REPO_ROOT / map_path, shutdown_dir / map_path)
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

    if method.key == "ours":
        pattern = "ply/voxel_model/iteration_*/voxel_model.ply"
    else:
        pattern = "ply/point_cloud/iteration_*/point_cloud.ply"
    return discover_one(
        shutdown_dir,
        pattern,
        f"{method.name} native map",
    )


def discover_run_artifacts(
    runs_root: Path,
    dataset: DatasetSpec,
    method: MethodSpec,
) -> RunArtifacts:
    run_dir = (
        runs_root / dataset.relative_run_dir / method.key
    ).resolve()
    shutdown_dir = discover_shutdown(run_dir)
    runtime_metrics = shutdown_dir / "runtime_metrics.json"
    runtime = load_json(runtime_metrics, "runtime metrics")
    trajectory = shutdown_dir / "CameraTrajectory_TUM.txt"
    require_file(trajectory, "run-local camera trajectory")
    native_map = resolve_native_map(runtime, shutdown_dir, method)

    surface_mesh: Path | None = None
    if dataset.key != "tum":
        if method.key == "ours":
            surface_pattern = (
                "ply/voxel_model/iteration_*/voxel_surface_mesh.ply"
            )
        else:
            surface_pattern = (
                "ply/point_cloud/iteration_*/gaussian_surface_mesh.ply"
            )
        surface_mesh = discover_one(
            shutdown_dir,
            surface_pattern,
            f"{method.name} surface mesh",
        )
    return RunArtifacts(
        dataset=dataset,
        method=method,
        run_dir=run_dir,
        shutdown_dir=shutdown_dir,
        runtime_metrics=runtime_metrics.resolve(),
        trajectory=trajectory.resolve(),
        native_map=native_map,
        surface_mesh=surface_mesh,
    )


def finite_float(value: Any, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid {field} in {source}: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {field} in {source}")
    return result


def positive_int(value: Any, field: str, source: Path) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid {field} in {source}: {value!r}"
        ) from error
    if result <= 0:
        raise RuntimeError(f"{field} must be positive in {source}")
    return result


def parse_tracking_times(path: Path, expected_frames: int) -> list[float]:
    require_file(path, "tracking-time file")
    values: list[float] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = finite_float(stripped, f"tracking time line {line_number}", path)
        if value > 0.0:
            values.append(value)
    if len(values) != expected_frames:
        raise RuntimeError(
            f"{path} contains {len(values)} positive frame times, "
            f"but runtime_metrics.json reports {expected_frames} frames"
        )
    return values


def parse_keyed_metric(path: Path) -> dict[int, float]:
    require_file(path, "per-keyframe metric file")
    values: dict[int, float] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise RuntimeError(
                f"Expected two columns at {path}:{line_number}"
            )
        try:
            keyframe_id = int(tokens[0])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe ID at {path}:{line_number}"
            ) from error
        value = finite_float(
            tokens[1],
            f"metric line {line_number}",
            path,
        )
        if keyframe_id in values:
            raise RuntimeError(
                f"Duplicate keyframe {keyframe_id} in {path}"
            )
        values[keyframe_id] = value
    if not values:
        raise RuntimeError(f"No metric values found in {path}")
    return values


def image_catalog(dataset: DatasetSpec) -> list[Path]:
    if dataset.key == "replica":
        image_dir = dataset.sequence_dir / "results"
        require_directory(image_dir, "Replica image directory")
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.name.startswith("frame")
        )
    else:
        assert dataset.image_list is not None
        require_file(dataset.image_list, f"{dataset.name} image list")
        images = []
        for line in dataset.image_list.read_text(
            encoding="utf-8"
        ).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                raise RuntimeError(
                    f"Malformed image-list line in {dataset.image_list}: "
                    f"{line}"
                )
            images.append(dataset.sequence_dir / tokens[1])
    if not images:
        raise RuntimeError(f"No dataset images found for {dataset.name}")
    return images


def dataset_basename_to_frame(dataset: DatasetSpec) -> dict[str, int]:
    result: dict[str, int] = {}
    for frame_id, path in enumerate(image_catalog(dataset)):
        basename = path.name
        if basename in result:
            raise RuntimeError(
                f"Dataset image basename is not unique: {basename}"
            )
        result[basename] = frame_id
    return result


def load_ours_keyframe_map(
    artifacts: RunArtifacts,
    dataset_frame_count: int,
) -> dict[int, int]:
    path = artifacts.shutdown_dir / "kf_frame_id_map.txt"
    require_file(path, "SVRecon keyframe/frame map")
    result: dict[int, int] = {}
    seen_frames: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise RuntimeError(
                f"Expected two columns at {path}:{line_number}"
            )
        try:
            keyframe_id, frame_id = (int(token) for token in tokens)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe map at {path}:{line_number}"
            ) from error
        if not 0 <= frame_id < dataset_frame_count:
            raise RuntimeError(
                f"Frame {frame_id} at {path}:{line_number} is outside "
                f"the dataset range"
            )
        if keyframe_id in result or frame_id in seen_frames:
            raise RuntimeError(
                f"Duplicate keyframe or dataset frame at "
                f"{path}:{line_number}"
            )
        result[keyframe_id] = frame_id
        seen_frames.add(frame_id)
    if not result:
        raise RuntimeError(f"No keyframe/frame entries found in {path}")
    return result


def load_photoslam_keyframe_map(
    artifacts: RunArtifacts,
    basename_to_frame: dict[str, int],
) -> dict[int, int]:
    cameras_path = artifacts.shutdown_dir / "ply/cameras.json"
    require_file(cameras_path, "Photo-SLAM cameras.json")
    try:
        cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot parse Photo-SLAM cameras: {cameras_path}"
        ) from error
    if not isinstance(cameras, list):
        raise RuntimeError(f"Expected a camera list in {cameras_path}")

    result: dict[int, int] = {}
    seen_frames: set[int] = set()
    for camera in cameras:
        if not isinstance(camera, dict):
            raise RuntimeError(f"Invalid camera entry in {cameras_path}")
        try:
            keyframe_id = int(camera["id"])
            image_name = str(camera["img_name"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Camera entry lacks a valid id/img_name in {cameras_path}"
            ) from error
        basename = Path(image_name).name
        if basename not in basename_to_frame:
            raise RuntimeError(
                f"Photo-SLAM image {image_name} is not an exact frame "
                f"from the evaluated dataset"
            )
        frame_id = basename_to_frame[basename]
        if keyframe_id in result or frame_id in seen_frames:
            raise RuntimeError(
                f"Duplicate keyframe or dataset frame in {cameras_path}"
            )
        result[keyframe_id] = frame_id
        seen_frames.add(frame_id)
    if not result:
        raise RuntimeError(f"No cameras found in {cameras_path}")
    return result


def keyframe_map(
    artifacts: RunArtifacts,
    basename_to_frame: dict[str, int],
) -> dict[int, int]:
    if artifacts.method.key == "ours":
        return load_ours_keyframe_map(
            artifacts,
            len(basename_to_frame),
        )
    return load_photoslam_keyframe_map(
        artifacts,
        basename_to_frame,
    )


def metric_by_dataset_frame(
    values: dict[int, float],
    keyframe_to_frame: dict[int, int],
    source: Path,
) -> dict[int, tuple[int, float]]:
    result: dict[int, tuple[int, float]] = {}
    missing = sorted(set(values).difference(keyframe_to_frame))
    if missing:
        raise RuntimeError(
            f"{source} contains keyframes absent from its frame map: "
            f"{missing[:10]}"
        )
    for keyframe_id, value in values.items():
        frame_id = keyframe_to_frame[keyframe_id]
        result[frame_id] = (keyframe_id, value)
    return result


def computational_result(artifacts: RunArtifacts) -> dict[str, Any]:
    runtime = load_json(
        artifacts.runtime_metrics,
        "runtime metrics",
    )
    frames = positive_int(
        runtime.get("frames"),
        "frames",
        artifacts.runtime_metrics,
    )
    keyframes = positive_int(
        runtime.get("keyframes"),
        "keyframes",
        artifacts.runtime_metrics,
    )
    iterations = positive_int(
        runtime.get("iterations"),
        "iterations",
        artifacts.runtime_metrics,
    )
    primitive_count = positive_int(
        runtime.get(
            "primitive_count",
            runtime.get(artifacts.method.representation),
        ),
        "primitive_count",
        artifacts.runtime_metrics,
    )
    mapping_seconds = finite_float(
        runtime.get("total_seconds"),
        "total_seconds",
        artifacts.runtime_metrics,
    )
    if mapping_seconds <= 0.0:
        raise RuntimeError(
            f"total_seconds must be positive in {artifacts.runtime_metrics}"
        )

    tracking_times = parse_tracking_times(
        artifacts.run_dir / "TrackingTime.txt",
        frames,
    )
    render_times = list(
        parse_keyed_metric(
            artifacts.shutdown_dir / "render_time.txt"
        ).values()
    )
    if any(value <= 0.0 for value in render_times):
        raise RuntimeError(
            f"Non-positive render time in "
            f"{artifacts.shutdown_dir / 'render_time.txt'}"
        )

    map_size_mb = (
        artifacts.native_map.stat().st_size / (1024.0 * 1024.0)
    )
    return {
        "dataset": artifacts.dataset.name,
        "method": artifacts.method.name,
        "family": artifacts.method.family,
        "frames": frames,
        "keyframes": keyframes,
        "primitive_type": artifacts.method.representation,
        "primitive_count": primitive_count,
        "iterations": iterations,
        "mapping_seconds": mapping_seconds,
        "system_fps_hz": frames / mapping_seconds,
        "tracking_fps_hz": len(tracking_times) / sum(tracking_times),
        "render_fps_hz": 1000.0 / statistics.fmean(render_times),
        "map_size_mb": map_size_mb,
        "gpu_memory_allocated_mb": finite_float(
            runtime.get("gpu_memory_allocated_mb"),
            "gpu_memory_allocated_mb",
            artifacts.runtime_metrics,
        ),
        "gpu_memory_reserved_mb": finite_float(
            runtime.get("gpu_memory_reserved_mb"),
            "gpu_memory_reserved_mb",
            artifacts.runtime_metrics,
        ),
        "runtime_scope": str(runtime.get("runtime_scope", "")),
        "run_dir": str(artifacts.run_dir),
        "shutdown_dir": str(artifacts.shutdown_dir),
        "native_map": str(artifacts.native_map),
    }


def photometric_results(
    dataset: DatasetSpec,
    artifacts_by_method: dict[str, RunArtifacts],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    basename_to_frame = dataset_basename_to_frame(dataset)
    method_values: dict[str, dict[int, dict[str, Any]]] = {}

    for method in METHODS:
        artifacts = artifacts_by_method[method.key]
        frame_map = keyframe_map(artifacts, basename_to_frame)
        psnr_path = artifacts.shutdown_dir / "psnr.txt"
        ssim_path = artifacts.shutdown_dir / "dssim.txt"
        render_path = artifacts.shutdown_dir / "render_time.txt"
        psnr = metric_by_dataset_frame(
            parse_keyed_metric(psnr_path),
            frame_map,
            psnr_path,
        )
        ssim = metric_by_dataset_frame(
            parse_keyed_metric(ssim_path),
            frame_map,
            ssim_path,
        )
        render = metric_by_dataset_frame(
            parse_keyed_metric(render_path),
            frame_map,
            render_path,
        )
        frame_ids = set(psnr) & set(ssim) & set(render)
        if (
            frame_ids != set(psnr)
            or frame_ids != set(ssim)
            or frame_ids != set(render)
        ):
            raise RuntimeError(
                f"Inconsistent photometric keys for "
                f"{dataset.name}/{method.name}"
            )
        method_values[method.key] = {
            frame_id: {
                "keyframe_id": psnr[frame_id][0],
                "psnr": psnr[frame_id][1],
                "ssim": ssim[frame_id][1],
                "render_ms": render[frame_id][1],
            }
            for frame_id in frame_ids
        }

    common_frames = set.intersection(
        *(set(method_values[method.key]) for method in METHODS)
    )
    if not common_frames:
        raise RuntimeError(
            f"No exact common rendered dataset frames for {dataset.name}"
        )

    summary: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [
            method_values[method.key][frame_id]
            for frame_id in sorted(common_frames)
        ]
        summary.append(
            {
                "dataset": dataset.name,
                "method": method.name,
                "family": method.family,
                "evaluated_frames": len(rows),
                "psnr": statistics.fmean(row["psnr"] for row in rows),
                "ssim": statistics.fmean(row["ssim"] for row in rows),
            }
        )
        for frame_id, row in zip(sorted(common_frames), rows):
            per_frame.append(
                {
                    "dataset": dataset.name,
                    "method": method.name,
                    "dataset_frame_id": frame_id,
                    **row,
                }
            )
    return summary, per_frame


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def markdown_table(
    rows: Iterable[dict[str, Any]],
    columns: list[tuple[str, str, str]],
) -> str:
    rows = list(rows)
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    separator = (
        "|" + "|".join("---:" if fmt else "---" for _, _, fmt in columns) + "|"
    )
    lines = [header, separator]
    for row in rows:
        rendered: list[str] = []
        for key, _, fmt in columns:
            value = row[key]
            rendered.append(format(value, fmt) if fmt else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_tabular_outputs(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    table: str,
) -> None:
    write_csv(output_dir / f"{stem}.csv", rows)
    write_json(
        output_dir / f"{stem}.json",
        {"protocol": protocol, "results": rows},
    )
    (output_dir / f"{stem}.txt").write_text(
        table + "\n",
        encoding="utf-8",
    )


def reconstruction_datasets(
    output_dir: Path,
    artifacts: dict[tuple[str, str], RunArtifacts],
    extra_replica_runs: tuple[ReconstructionExperiment, ...] = (),
) -> tuple[ReconstructionDataset, ...]:
    replica_runs = tuple(
        ReconstructionExperiment(
            name=method.name,
            family=method.family,
            directory=(
                output_dir
                / "reconstruction/replica"
                / method.key
            ),
            mesh=artifacts[("replica", method.key)].surface_mesh,
            recon_trajectory=(
                artifacts[("replica", method.key)].trajectory
            ),
        )
        for method in METHODS
    ) + extra_replica_runs
    scannet_runs = tuple(
        ReconstructionExperiment(
            name=method.name,
            family=method.family,
            directory=(
                output_dir
                / "reconstruction/scannet"
                / method.key
            ),
            mesh=artifacts[("scannet", method.key)].surface_mesh,
            recon_trajectory=(
                artifacts[("scannet", method.key)].trajectory
            ),
        )
        for method in METHODS
    )

    replica_gt_trajectory = (
        REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
    )
    placeholder_trajectory = replica_runs[0].recon_trajectory
    assert placeholder_trajectory is not None
    return (
        ReconstructionDataset(
            name="Replica office0 (culled GT)",
            gt_mesh=(
                REPO_ROOT
                / "third_party/ESLAM/cull_replica_mesh/office0_culled.ply"
            ),
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=placeholder_trajectory,
            artifact_tag="culled",
            summary_dir=(
                output_dir / "reconstruction/replica_culled"
            ),
            experiments=replica_runs,
        ),
        ReconstructionDataset(
            name="Replica office0 (original GT)",
            gt_mesh=(
                REPO_ROOT / "scripts/data/Replica/office0_mesh.ply"
            ),
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=placeholder_trajectory,
            artifact_tag="original",
            summary_dir=(
                output_dir / "reconstruction/replica_original"
            ),
            experiments=replica_runs,
        ),
        ReconstructionDataset(
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
            shared_recon_trajectory=(
                scannet_runs[0].recon_trajectory
                or placeholder_trajectory
            ),
            summary_dir=(
                output_dir / "reconstruction/scannet"
            ),
            experiments=scannet_runs,
        ),
    )


def evaluate_reconstruction(
    output_dir: Path,
    artifacts: dict[tuple[str, str], RunArtifacts],
    extra_replica_runs: tuple[ReconstructionExperiment, ...] = (),
) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for dataset in reconstruction_datasets(
        output_dir,
        artifacts,
        extra_replica_runs,
    ):
        print(f"\n=== Reconstruction: {dataset.name} ===")
        require_file(dataset.gt_mesh, f"{dataset.name} GT mesh")
        gt_trajectory = materialize_gt_trajectory(dataset)
        dataset_results: list[dict[str, Any]] = []
        for experiment in dataset.experiments:
            assert experiment.mesh is not None
            assert experiment.recon_trajectory is not None
            experiment.directory.mkdir(parents=True, exist_ok=True)
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
                experiment.mesh,
                experiment.recon_trajectory,
                alignment_gt_trajectory,
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
            aggregate.append(result)
        write_summaries(
            dataset,
            gt_trajectory,
            dataset_results,
            DISTANCE_THRESHOLD_M,
        )
    return aggregate


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    require_directory(runs_root, "monocular benchmark root")
    output_dir = runs_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[tuple[str, str], RunArtifacts] = {}
    for dataset in DATASETS:
        for method in METHODS:
            artifacts[(dataset.key, method.key)] = discover_run_artifacts(
                runs_root,
                dataset,
                method,
            )

    computational = [
        computational_result(artifacts[(dataset.key, method.key)])
        for dataset in DATASETS
        for method in METHODS
    ]
    computational_table = markdown_table(
        computational,
        [
            ("dataset", "Dataset", ""),
            ("method", "Method", ""),
            ("system_fps_hz", "System FPS", ".2f"),
            ("tracking_fps_hz", "Tracking FPS", ".2f"),
            ("render_fps_hz", "Render FPS", ".2f"),
            ("map_size_mb", "Map MB", ".2f"),
            ("gpu_memory_reserved_mb", "GPU reserved MB", ".2f"),
        ],
    )
    write_tabular_outputs(
        output_dir,
        "computational_metrics",
        computational,
        {
            "system_runtime": (
                "mapping and tail optimization; shutdown evaluation "
                "rendering and mesh export excluded"
            ),
            "tracking_runtime": "ORB-SLAM TrackMonocular calls only",
            "map_size": "native voxel_model.ply or point_cloud.ply",
            "gpu_memory": "PyTorch CUDA caching allocator peak",
        },
        computational_table,
    )

    photometric: list[dict[str, Any]] = []
    photometric_per_frame: list[dict[str, Any]] = []
    for dataset in DATASETS:
        summary, per_frame = photometric_results(
            dataset,
            {
                method.key: artifacts[(dataset.key, method.key)]
                for method in METHODS
            },
        )
        photometric.extend(summary)
        photometric_per_frame.extend(per_frame)
    photometric_table = markdown_table(
        photometric,
        [
            ("dataset", "Dataset", ""),
            ("method", "Method", ""),
            ("evaluated_frames", "Common frames", "d"),
            ("psnr", "PSNR", ".4f"),
            ("ssim", "SSIM", ".4f"),
        ],
    )
    write_tabular_outputs(
        output_dir,
        "photometric_metrics",
        photometric,
        {
            "views": (
                "intersection of exact dataset frame IDs rendered by "
                "both methods"
            ),
            "psnr_source": "shutdown psnr.txt, computed before JPEG export",
            "ssim_source": (
                "shutdown dssim.txt; despite its legacy name, the stored "
                "quantity is SSIM"
            ),
        },
        photometric_table,
    )
    write_csv(
        output_dir / "photometric_metrics_per_frame.csv",
        photometric_per_frame,
    )

    extra_replica_runs: tuple[ReconstructionExperiment, ...] = ()
    if args.monogs_replica_run is not None:
        extra_replica_runs = (
            prepare_monogs_replica_experiment(
                args.monogs_replica_run,
            ),
        )
    reconstruction = evaluate_reconstruction(
        output_dir,
        artifacts,
        extra_replica_runs,
    )
    metric_suffix = threshold_label(DISTANCE_THRESHOLD_M)
    reconstruction_table = markdown_table(
        reconstruction,
        [
            ("dataset", "Dataset", ""),
            ("name", "Method", ""),
            ("accuracy_m", "Acc. m", ".6f"),
            ("completeness_m", "Compl. m", ".6f"),
            ("chamfer_l1_m", "Chamfer m", ".6f"),
            (f"precision_{metric_suffix}", "Prec@5cm", ".4f"),
            (f"completion_{metric_suffix}", "Recall@5cm", ".4f"),
            (f"fscore_{metric_suffix}", "F@5cm", ".4f"),
        ],
    )
    write_tabular_outputs(
        output_dir,
        "reconstruction_metrics",
        reconstruction,
        {
            "tum": "not evaluated because no GT reconstruction is provided",
            "alignment": "trajectory Sim(3), then HI-SLAM2 rigid ICP",
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "replica_gt": "culled and original meshes evaluated separately",
            "scannet_gt": "scene0000_00_vh_clean_2.ply",
        },
        reconstruction_table,
    )

    summary = (
        "# Monocular Benchmark\n\n"
        "## Computational\n\n"
        f"{computational_table}\n\n"
        "## Photometric\n\n"
        f"{photometric_table}\n\n"
        "## Reconstruction\n\n"
        f"{reconstruction_table}\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(f"\nSaved monocular evaluation to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
