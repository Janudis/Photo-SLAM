#!/usr/bin/env python3
"""Evaluate one completed monocular SVRecon run on EuRoC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SHUTDOWN_PATTERN = re.compile(r"^\d+_shutdown$")
SEQUENCE_GT = {
    "MH_01_easy": "MH01_GT.txt",
    "MH_02_easy": "MH02_GT.txt",
    "V1_01_easy": "V101_GT.txt",
    "V2_01_easy": "V201_GT.txt",
}
DEFAULT_EVO_APE = (
    Path.home() / "miniconda3/envs/MonoGS/bin/evo_ape"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EuRoC trajectory, photometric quality, runtime, "
            "memory, and map size for a completed monocular SVRecon run."
        )
    )
    parser.add_argument(
        "--sequence",
        choices=tuple(SEQUENCE_GT),
        default="MH_01_easy",
        help="EuRoC sequence (default: MH_01_easy)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "run root or numeric *_shutdown directory; defaults to "
            "results/euroc_voxel/<sequence>"
        ),
    )
    parser.add_argument(
        "--evo-ape",
        type=Path,
        default=DEFAULT_EVO_APE,
        help=f"evo_ape executable (default: {DEFAULT_EVO_APE})",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def discover_shutdown(run_dir: Path) -> tuple[Path, Path]:
    run_dir = run_dir.expanduser().resolve()
    if SHUTDOWN_PATTERN.fullmatch(run_dir.name):
        require_directory(run_dir, "shutdown directory")
        return run_dir.parent, run_dir

    require_directory(run_dir, "EuRoC run directory")
    candidates = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and SHUTDOWN_PATTERN.fullmatch(path.name)
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Expected exactly one numeric *_shutdown directory in "
            f"{run_dir}; found {len(candidates)}"
            + (f": {names}" if names else "")
        )
    return run_dir, candidates[0].resolve()


def load_json(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return value


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


def parse_tracking_times(path: Path) -> list[float]:
    require_file(path, "tracking-time file")
    values: list[float] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = finite_float(
            stripped,
            f"tracking time line {line_number}",
            path,
        )
        if value <= 0.0:
            raise RuntimeError(
                f"Tracking time must be positive at {path}:{line_number}"
            )
        values.append(value)
    if not values:
        raise RuntimeError(f"No tracking times found in {path}")
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
        if keyframe_id in values:
            raise RuntimeError(
                f"Duplicate keyframe {keyframe_id} in {path}"
            )
        values[keyframe_id] = finite_float(
            tokens[1],
            f"metric line {line_number}",
            path,
        )
    if not values:
        raise RuntimeError(f"No metric values found in {path}")
    return values


def resolve_native_map(
    runtime: dict[str, Any],
    runtime_path: Path,
    shutdown_dir: Path,
) -> Path:
    map_value = runtime.get("map_path")
    if isinstance(map_value, str) and map_value:
        configured = Path(map_value)
        candidates = [configured]
        if not configured.is_absolute():
            candidates.extend(
                (REPO_ROOT / configured, shutdown_dir / configured)
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

    candidates = sorted(
        shutdown_dir.glob(
            "ply/voxel_model/iteration_*/voxel_model.ply"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not resolve one native voxel map from "
            f"{runtime_path} or {shutdown_dir}"
        )
    return candidates[0].resolve()


def resolve_surface_mesh(shutdown_dir: Path) -> Path | None:
    candidates = sorted(
        shutdown_dir.glob(
            "ply/voxel_model/iteration_*/voxel_surface_mesh.ply"
        )
    )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected at most one surface mesh below {shutdown_dir}"
        )
    return candidates[0].resolve() if candidates else None


def evaluate_trajectory(
    sequence: str,
    trajectory: Path,
    evo_ape: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require_file(evo_ape, "evo_ape executable")
    gt_euroc = require_file(
        REPO_ROOT
        / "ORB-SLAM3/evaluation/Ground_truth/EuRoC_left_cam"
        / SEQUENCE_GT[sequence],
        "EuRoC left-camera ground-truth trajectory",
    )
    require_file(trajectory, "estimated TUM trajectory")

    gt_tum = output_dir / "ground_truth_left_camera_tum.txt"
    result_zip = output_dir / "trajectory_ape.zip"
    log_path = output_dir / "trajectory_ape.txt"
    for path in (gt_tum, result_zip, log_path):
        if path.exists():
            path.unlink()
    materialize_left_camera_gt_tum(gt_euroc, gt_tum)

    command = [
        str(evo_ape),
        "tum",
        str(gt_tum),
        str(trajectory),
        "-r",
        "trans_part",
        "-a",
        "-s",
        "--t_max_diff",
        "0.01",
        "--no_warnings",
        "--save_results",
        str(result_zip),
    ]
    print("[EuRoC trajectory] " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")

    require_file(result_zip, "evo APE result archive")
    with zipfile.ZipFile(result_zip) as archive:
        stats = json.loads(
            archive.read("stats.json").decode("utf-8")
        )
        info = json.loads(
            archive.read("info.json").decode("utf-8")
        )
    if not isinstance(stats, dict) or not isinstance(info, dict):
        raise RuntimeError(f"Invalid evo result archive: {result_zip}")

    result = {
        "alignment": "Sim(3) Umeyama",
        "pose_relation": "translation part",
        "timestamp_max_difference_s": 0.01,
        "ground_truth_euroc_left_camera": str(gt_euroc),
        "ground_truth_tum": str(gt_tum),
        "estimate": str(trajectory),
        "result_archive": str(result_zip),
        "log": str(log_path),
    }
    result.update(
        {
            f"ate_{key}_m": finite_float(
                value,
                f"evo {key}",
                result_zip,
            )
            for key, value in stats.items()
        }
    )
    if "num_pairs" in info:
        result["associated_poses"] = int(info["num_pairs"])
    return result


def write_per_keyframe(
    path: Path,
    frame_ids: list[int],
    psnr: dict[int, float],
    ssim: dict[int, float],
    render: dict[int, float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "keyframe_id",
                "psnr",
                "ssim",
                "render_time_ms",
            ),
        )
        writer.writeheader()
        for keyframe_id in frame_ids:
            writer.writerow(
                {
                    "keyframe_id": keyframe_id,
                    "psnr": psnr[keyframe_id],
                    "ssim": ssim[keyframe_id],
                    "render_time_ms": render[keyframe_id],
                }
            )


def materialize_left_camera_gt_tum(
    source: Path,
    destination: Path,
) -> None:
    require_file(source, "EuRoC left-camera ground-truth trajectory")
    rows: list[str] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = [token.strip() for token in stripped.split(",")]
        if len(tokens) != 8:
            raise RuntimeError(
                f"Expected eight columns at {source}:{line_number}"
            )
        try:
            timestamp_ns = int(tokens[0].split(".", maxsplit=1)[0])
            values = [float(token) for token in tokens[1:]]
        except ValueError as error:
            raise RuntimeError(
                f"Invalid trajectory value at {source}:{line_number}"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Non-finite trajectory value at {source}:{line_number}"
            )
        x, y, z, qw, qx, qy, qz = values
        rows.append(
            f"{timestamp_ns / 1.0e9:.9f} "
            f"{x:.10f} {y:.10f} {z:.10f} "
            f"{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}"
        )
    if not rows:
        raise RuntimeError(f"No trajectory poses found in {source}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    default_run_dir = (
        REPO_ROOT / "results/euroc_voxel" / args.sequence
    )
    run_dir, shutdown_dir = discover_shutdown(
        args.run_dir if args.run_dir is not None else default_run_dir
    )
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = shutdown_dir / "runtime_metrics.json"
    runtime = load_json(runtime_path, "runtime metrics")
    frames = positive_int(runtime.get("frames"), "frames", runtime_path)
    keyframes = positive_int(
        runtime.get("keyframes"),
        "keyframes",
        runtime_path,
    )
    voxels = positive_int(
        runtime.get("primitive_count", runtime.get("voxels")),
        "primitive_count",
        runtime_path,
    )
    iterations = positive_int(
        runtime.get("iterations"),
        "iterations",
        runtime_path,
    )
    mapping_seconds = finite_float(
        runtime.get("total_seconds"),
        "total_seconds",
        runtime_path,
    )
    if mapping_seconds <= 0.0:
        raise RuntimeError(
            f"total_seconds must be positive in {runtime_path}"
        )

    tracking_path = shutdown_dir / "TrackingTime.txt"
    if not tracking_path.is_file():
        tracking_path = run_dir / "TrackingTime.txt"
    tracking_times = parse_tracking_times(tracking_path)
    if len(tracking_times) != frames:
        raise RuntimeError(
            f"{tracking_path} has {len(tracking_times)} frames, but "
            f"{runtime_path} reports {frames}"
        )

    psnr = parse_keyed_metric(shutdown_dir / "psnr.txt")
    ssim = parse_keyed_metric(shutdown_dir / "dssim.txt")
    render = parse_keyed_metric(shutdown_dir / "render_time.txt")
    common_keyframes = sorted(set(psnr) & set(ssim) & set(render))
    if (
        not common_keyframes
        or set(common_keyframes) != set(psnr)
        or set(common_keyframes) != set(ssim)
        or set(common_keyframes) != set(render)
    ):
        raise RuntimeError(
            "PSNR, SSIM, and render-time files do not contain the "
            "same non-empty keyframe set"
        )
    if any(render[key] <= 0.0 for key in common_keyframes):
        raise RuntimeError("Render times must be positive")

    native_map = resolve_native_map(runtime, runtime_path, shutdown_dir)
    surface_mesh = resolve_surface_mesh(shutdown_dir)
    trajectory = shutdown_dir / "CameraTrajectory_TUM.txt"
    trajectory_result = evaluate_trajectory(
        args.sequence,
        trajectory,
        args.evo_ape.expanduser().resolve(),
        output_dir,
    )

    computational = {
        "frames": frames,
        "keyframes": keyframes,
        "voxels": voxels,
        "iterations": iterations,
        "mapping_seconds": mapping_seconds,
        "system_fps_hz": frames / mapping_seconds,
        "tracking_fps_hz": (
            len(tracking_times) / sum(tracking_times)
        ),
        "render_fps_hz": (
            1000.0
            / statistics.fmean(
                render[key] for key in common_keyframes
            )
        ),
        "map_size_mb": native_map.stat().st_size / (1024.0 * 1024.0),
        "gpu_memory_allocated_mb": finite_float(
            runtime.get("gpu_memory_allocated_mb"),
            "gpu_memory_allocated_mb",
            runtime_path,
        ),
        "gpu_memory_reserved_mb": finite_float(
            runtime.get("gpu_memory_reserved_mb"),
            "gpu_memory_reserved_mb",
            runtime_path,
        ),
        "runtime_scope": str(runtime.get("runtime_scope", "")),
    }
    photometric = {
        "evaluated_keyframes": len(common_keyframes),
        "psnr": statistics.fmean(
            psnr[key] for key in common_keyframes
        ),
        "ssim": statistics.fmean(
            ssim[key] for key in common_keyframes
        ),
    }
    result = {
        "sequence": args.sequence,
        "run_dir": str(run_dir),
        "shutdown_dir": str(shutdown_dir),
        "native_map": str(native_map),
        "surface_mesh": str(surface_mesh) if surface_mesh else None,
        "computational": computational,
        "photometric": photometric,
        "trajectory": trajectory_result,
        "reconstruction": {
            "quantitative_metrics_available": False,
            "reason": (
                "EuRoC does not provide a comparable dense "
                "ground-truth surface mesh."
            ),
        },
    }

    write_per_keyframe(
        output_dir / "photometric_metrics_per_keyframe.csv",
        common_keyframes,
        psnr,
        ssim,
        render,
    )
    (output_dir / "euroc_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ate_rmse = trajectory_result.get("ate_rmse_m")
    if ate_rmse is None:
        raise RuntimeError("evo result contains no RMSE statistic")
    text = (
        f"EuRoC sequence: {args.sequence}\n"
        f"ATE RMSE, Sim(3): {ate_rmse:.6f} m\n"
        f"PSNR: {photometric['psnr']:.4f}\n"
        f"SSIM: {photometric['ssim']:.4f}\n"
        f"Evaluated keyframes: {photometric['evaluated_keyframes']}\n"
        f"System FPS: {computational['system_fps_hz']:.4f}\n"
        f"Tracking FPS: {computational['tracking_fps_hz']:.4f}\n"
        f"Render FPS: {computational['render_fps_hz']:.4f}\n"
        f"Map size: {computational['map_size_mb']:.4f} MB\n"
        f"GPU reserved: "
        f"{computational['gpu_memory_reserved_mb']:.2f} MB\n"
        f"Voxels: {voxels}\n"
        "Dense reconstruction metrics: N/A (no comparable EuRoC "
        "GT surface mesh)\n"
    )
    (output_dir / "euroc_metrics.txt").write_text(
        text,
        encoding="utf-8",
    )
    print("\n" + text)
    print(f"Saved EuRoC evaluation to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
