#!/usr/bin/env python3
"""Evaluate completed native paper benchmarks.

The Replica path intentionally follows the HI-SLAM2 reconstruction protocol:
trajectory Sim(3) alignment, rigid ICP in ``eval_recon.py``, and 3D metrics at
the requested distance threshold. Appearance metrics use every keyframe
rendered by each completed mapping run, matching the native mapper outputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ours", "ours_mvs", "photoslam")
METHOD_LABELS = {
    "ours": "Ours",
    "ours_mvs": "Ours+MVS",
    "photoslam": "Photo-SLAM",
}
REPLICA_SEQUENCES = (
    "room0",
    "room1",
    "room2",
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NUMPY_SCALAR_PATTERN = re.compile(
    r"\b(?:np|numpy)\.(?:float(?:16|32|64|128)?|"
    r"int(?:8|16|32|64)?|uint(?:8|16|32|64)?)\s*\("
)


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    method: str
    sequence: str
    trial: int
    shutdown_dir: Path


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise EvaluationError(f"Missing {description}: {path}")
    return path


def require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise EvaluationError(f"Missing {description}: {path}")
    return path


def load_json(path: Path, description: str) -> Any:
    require_file(path, description)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot parse {description}: {path}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    rendered = " ".join(command)
    print(f"[cwd={cwd}] $ {rendered}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
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
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise EvaluationError(
            f"Command exited with status {return_code}; see {log_path}"
        )


def resolve_manifest_path(path_value: str, run_root: Path, job: dict[str, Any]) -> Path:
    path = Path(path_value)
    if path.exists():
        return path.resolve()
    fallback = (
        run_root
        / "replica"
        / str(job["sequence"])
        / str(job["method"])
        / f"trial_{int(job['trial']):02d}"
        / path.name
    )
    if fallback.exists():
        return fallback.resolve()
    return path


def load_jobs(
    run_root: Path,
    expected_methods: tuple[str, ...] = METHODS,
) -> list[Job]:
    manifest = load_json(run_root / "run_manifest.json", "run manifest")
    if manifest.get("status") != "complete":
        raise EvaluationError(
            f"Benchmark is not complete: {manifest.get('status')!r}"
        )
    if manifest.get("dataset") != "replica":
        raise EvaluationError("Only Replica evaluation is currently supported")
    jobs: list[Job] = []
    for raw in manifest.get("jobs", []):
        if raw.get("status") != "complete":
            raise EvaluationError(f"Incomplete benchmark job: {raw}")
        method = str(raw["method"])
        sequence = str(raw["sequence"])
        trial = int(raw["trial"])
        shutdown = resolve_manifest_path(
            str(raw["shutdown_dir"]), run_root, raw
        )
        require_directory(shutdown, "shutdown directory")
        jobs.append(Job(method, sequence, trial, shutdown))

    expected = {
        (method, sequence, 1)
        for method in expected_methods
        for sequence in REPLICA_SEQUENCES
    }
    actual = {(job.method, job.sequence, job.trial) for job in jobs}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvaluationError(
            f"Expected one 24-job Replica matrix; missing={missing}, extra={extra}"
        )
    return jobs


def discover_latest(pattern: str, root: Path, description: str) -> Path:
    candidates = list(root.glob(pattern))
    if not candidates:
        raise EvaluationError(f"No {description} found below {root}")

    def iteration(path: Path) -> int:
        match = re.fullmatch(r"iteration_(\d+)", path.parent.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=iteration).resolve()


def raw_mesh(job: Job) -> Path:
    if job.method in {"ours", "ours_mvs"}:
        return discover_latest(
            "ply/voxel_model/iteration_*/voxel_surface_mesh.ply",
            job.shutdown_dir,
            "SVRecon surface mesh",
        )
    return discover_latest(
        "ply/point_cloud/iteration_*/gaussian_surface_mesh.ply",
        job.shutdown_dir,
        "Photo-SLAM Gaussian surface mesh",
    )


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
            raise EvaluationError(f"Expected two columns at {path}:{line_number}")
        keyframe = int(tokens[0])
        value = float(tokens[1])
        if keyframe in values or not math.isfinite(value):
            raise EvaluationError(f"Invalid metric at {path}:{line_number}")
        values[keyframe] = value
    if not values:
        raise EvaluationError(f"No metrics found in {path}")
    return values


def replica_images(sequence_dir: Path) -> list[Path]:
    result_dir = require_directory(sequence_dir / "results", "Replica images")
    images = sorted(
        path
        for path in result_dir.iterdir()
        if path.is_file()
        and path.name.startswith("frame")
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise EvaluationError(f"No Replica RGB images found in {result_dir}")
    return images


def ours_keyframes(job: Job, frame_count: int) -> dict[int, int]:
    path = require_file(
        job.shutdown_dir / "kf_frame_id_map.txt", "keyframe/frame map"
    )
    result: dict[int, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise EvaluationError(f"Expected two columns at {path}:{line_number}")
        keyframe, frame = map(int, tokens)
        if not 0 <= frame < frame_count or keyframe in result:
            raise EvaluationError(f"Invalid keyframe map at {path}:{line_number}")
        result[keyframe] = frame
    return result


def photoslam_keyframes(job: Job, images: list[Path]) -> dict[int, int]:
    cameras = load_json(job.shutdown_dir / "ply/cameras.json", "cameras.json")
    by_name = {path.name: index for index, path in enumerate(images)}
    result: dict[int, int] = {}
    for camera in cameras:
        keyframe = int(camera["id"])
        name = Path(str(camera["img_name"])).name
        if name not in by_name:
            raise EvaluationError(f"Camera image is not a Replica frame: {name}")
        if keyframe in result:
            raise EvaluationError(f"Duplicate Photo-SLAM keyframe {keyframe}")
        result[keyframe] = by_name[name]
    return result


def keyframe_map(job: Job, images: list[Path]) -> dict[int, int]:
    if job.method in {"ours", "ours_mvs"}:
        return ours_keyframes(job, len(images))
    return photoslam_keyframes(job, images)


def metric_by_frame(job: Job, images: list[Path]) -> dict[int, dict[str, Any]]:
    mapping = keyframe_map(job, images)
    psnr = parse_keyed_metric(job.shutdown_dir / "psnr.txt")
    ssim = parse_keyed_metric(job.shutdown_dir / "dssim.txt")
    if set(psnr) != set(ssim):
        raise EvaluationError(f"PSNR/SSIM keys differ in {job.shutdown_dir}")
    missing = set(psnr) - set(mapping)
    if missing:
        raise EvaluationError(
            f"Metric keyframes are absent from frame map: {sorted(missing)[:10]}"
        )
    result: dict[int, dict[str, Any]] = {}
    for keyframe in psnr:
        frame = mapping[keyframe]
        if frame in result:
            raise EvaluationError(f"Duplicate dataset frame {frame} in {job.shutdown_dir}")
        result[frame] = {
            "keyframe_id": keyframe,
            "psnr": psnr[keyframe],
            "ssim": ssim[keyframe],
        }
    return result


def rendered_image_index(job: Job) -> dict[int, Path]:
    image_dir = require_directory(job.shutdown_dir / "image", "rendered images")
    result: dict[int, Path] = {}
    pattern = re.compile(r"^\d+_(\d+)(?:_shutdown)?$")
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = pattern.fullmatch(path.stem)
        if not match:
            continue
        keyframe = int(match.group(1))
        if keyframe in result:
            raise EvaluationError(f"Duplicate render for keyframe {keyframe}: {path}")
        result[keyframe] = path
    if not result:
        raise EvaluationError(f"No rendered images found in {image_dir}")
    return result


def load_lpips(device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        import lpips
        from PIL import Image
    except ImportError as error:
        raise EvaluationError(
            "LPIPS dependencies are missing. Install lpips and Pillow in the "
            "benchmark container."
        ) from error
    if device == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA was requested for LPIPS but is unavailable")
    model = lpips.LPIPS(net="alex").to(device).eval()
    return torch, Image, model


def image_tensor(torch: Any, Image: Any, path: Path, size: tuple[int, int] | None) -> Any:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def lpips_for_frames(
    job: Job,
    images: list[Path],
    metrics: dict[int, dict[str, Any]],
    frame_ids: list[int],
    device: str,
    cache_path: Path,
    runtime: tuple[Any, Any, Any],
) -> dict[int, float]:
    if cache_path.is_file():
        cached = load_json(cache_path, "LPIPS cache")
        values = {int(key): float(value) for key, value in cached.items()}
        if set(values) == set(frame_ids):
            return values

    torch, Image, model = runtime
    rendered = rendered_image_index(job)
    values: dict[int, float] = {}
    with torch.inference_mode():
        for index, frame in enumerate(frame_ids, start=1):
            keyframe = int(metrics[frame]["keyframe_id"])
            if keyframe not in rendered:
                raise EvaluationError(
                    f"No rendered image for keyframe {keyframe} in {job.shutdown_dir}"
                )
            render_path = rendered[keyframe]
            render_pil = Image.open(render_path).convert("RGB")
            size = render_pil.size
            render_array = np.asarray(render_pil, dtype=np.float32) / 127.5 - 1.0
            render_tensor = (
                torch.from_numpy(render_array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            gt_tensor = image_tensor(torch, Image, images[frame], size).to(device)
            values[frame] = float(model(render_tensor, gt_tensor).item())
            if index % 25 == 0 or index == len(frame_ids):
                print(
                    f"[LPIPS] {job.sequence}/{job.method}: "
                    f"{index}/{len(frame_ids)}",
                    flush=True,
                )
    write_json(cache_path, {str(key): value for key, value in values.items()})
    return values


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    # Stable branch form returning TUM's qx qy qz qw convention.
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def materialize_replica_tum(gt_path: Path, output_path: Path, fps: float = 30.0) -> None:
    lines: list[str] = []
    for index, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        values = np.fromstring(stripped, sep=" ", dtype=np.float64)
        if values.size != 16 or not np.all(np.isfinite(values)):
            raise EvaluationError(f"Invalid Replica pose at {gt_path}:{index + 1}")
        pose = values.reshape(4, 4)
        qx, qy, qz, qw = rotation_to_quaternion(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        timestamp = index / fps
        lines.append(
            f"{timestamp:.9f} {tx:.12g} {ty:.12g} {tz:.12g} "
            f"{qx:.12g} {qy:.12g} {qz:.12g} {qw:.12g}"
        )
    if not lines:
        raise EvaluationError(f"No Replica poses found in {gt_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_ate(
    job: Job,
    gt_trajectory: Path,
    output_dir: Path,
    evo_ape: str,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_tum = output_dir / "gt_trajectory_tum.txt"
    materialize_replica_tum(gt_trajectory, gt_tum)
    estimate = require_file(
        job.shutdown_dir / "CameraTrajectory_TUM.txt", "estimated trajectory"
    )
    log_path = output_dir / "evo_ape.txt"
    result_zip = output_dir / "evo_ape.zip"
    if log_path.is_file():
        existing = log_path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*rmse\s+([0-9.eE+-]+)\s*$", existing)
        if match:
            return 100.0 * float(match.group(1))
    if result_zip.is_file():
        result_zip.unlink()
    command = [
        evo_ape,
        "tum",
        str(gt_tum),
        str(estimate),
        "-vas",
        "--save_results",
        str(result_zip),
        "--no_warnings",
    ]
    run_logged(command, REPO_ROOT, log_path)
    text = log_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*rmse\s+([0-9.eE+-]+)\s*$", text)
    if not match:
        raise EvaluationError(f"Cannot parse ATE RMSE from {log_path}")
    return 100.0 * float(match.group(1))


def parse_hi_metrics(path: Path) -> dict[str, float]:
    serialized = NUMPY_SCALAR_PATTERN.sub("(", path.read_text(encoding="utf-8"))
    try:
        raw = ast.literal_eval(serialized)
    except (SyntaxError, ValueError) as error:
        raise EvaluationError(f"Cannot parse HI-SLAM2 metrics: {path}") from error
    required = {
        "mean precision",
        "mean recall",
        "precision",
        "recall",
        "f-score",
    }
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise EvaluationError(f"Incomplete HI-SLAM2 metrics: {path}")
    return {key: float(raw[key]) for key in required}


def evaluate_geometry(
    job: Job,
    gt_mesh: Path,
    gt_trajectory: Path,
    output_dir: Path,
    threshold_m: float,
    hislam_python: str,
    force: bool,
) -> dict[str, float]:
    metrics_path = output_dir / "eval_recon_hi_official.txt"
    if metrics_path.is_file() and not force:
        return parse_hi_metrics(metrics_path)

    mesh_eval = require_file(REPO_ROOT / "bin/mesh_eval", "mesh_eval executable")
    evaluator = require_file(
        REPO_ROOT / "third_party/HI-SLAM2/scripts/eval_recon.py",
        "HI-SLAM2 evaluator",
    )
    alignment_dir = output_dir / "alignment"
    aligned_mesh = alignment_dir / "recon_mesh_aligned.ply"
    alignment_command = [
        str(mesh_eval),
        "--eval_mode=current",
        f"--recon={raw_mesh(job)}",
        f"--gt={gt_mesh}",
        f"--out={alignment_dir}",
        f"--tau_cm={100.0 * threshold_m:g}",
        "--align_recon_to_gt=1",
        f"--traj={gt_trajectory}",
        "--traj_mode=c2w",
        f"--recon_traj_tum={job.shutdown_dir / 'CameraTrajectory_TUM.txt'}",
        "--save_aligned_mesh=1",
        "--alignment_only=1",
        "--eval_floaters=0",
        "--eval_gaussian_support=0",
        "--eval_voxel_support=0",
    ]
    run_logged(alignment_command, REPO_ROOT, output_dir / "alignment.log")
    require_file(aligned_mesh, "aligned reconstruction mesh")

    eval_command = [
        hislam_python,
        str(evaluator),
        str(aligned_mesh),
        str(gt_mesh),
        "--eval_3d",
        "--distance_thresh",
        f"{threshold_m:.12g}",
        "--save",
        str(metrics_path),
    ]
    run_logged(eval_command, evaluator.parent, output_dir / "eval_recon.log")
    return parse_hi_metrics(metrics_path)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise EvaluationError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def aggregate(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for method in METHODS:
        rows = [results[sequence][method] for sequence in REPLICA_SEQUENCES]
        keys = [
            "ate_rmse_cm",
            "accuracy_cm",
            "completeness_cm",
            "completion_ratio_percent",
            "precision_5cm",
            "recall_5cm",
            "fscore_5cm",
            "psnr",
            "ssim",
            "lpips",
        ]
        aggregated[method] = {
            key: mean(float(row[key]) for row in rows) for key in keys
        }
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a complete native Replica paper benchmark."
    )
    parser.add_argument("dataset", choices=("replica",))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--photoslam-run-id",
        help=(
            "optional Photo-SLAM-only run containing shutdown mesh exports; "
            "Ours and Ours+MVS remain sourced from --run-id"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("PHOTOSLAM_DATA_ROOT", REPO_ROOT / "scripts/data")),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("PHOTOSLAM_RESULTS_ROOT", REPO_ROOT / "results")),
    )
    parser.add_argument(
        "--gt-mesh-root",
        type=Path,
        default=REPO_ROOT / "third_party/ESLAM/cull_replica_mesh",
        help="HI-SLAM2 Replica GT mesh directory",
    )
    parser.add_argument("--distance-threshold-m", type=float, default=0.05)
    parser.add_argument("--hislam-python", default=sys.executable)
    parser.add_argument("--evo-ape", default=shutil.which("evo_ape") or "evo_ape")
    parser.add_argument("--lpips-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--force-reconstruction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.distance_threshold_m <= 0.0 or not math.isfinite(args.distance_threshold_m):
        raise EvaluationError("Distance threshold must be positive and finite")
    run_root = args.results_root / "paper_benchmark" / args.run_id
    jobs = load_jobs(run_root)
    if args.photoslam_run_id:
        photoslam_root = (
            args.results_root / "paper_benchmark" / args.photoslam_run_id
        )
        replacement = load_jobs(photoslam_root, ("photoslam",))
        jobs = [job for job in jobs if job.method != "photoslam"] + replacement
    by_pair = {(job.sequence, job.method): job for job in jobs}
    output_root = run_root / "evaluation"
    results: dict[str, dict[str, dict[str, Any]]] = {}
    lpips_runtime = None if args.skip_lpips else load_lpips(args.lpips_device)

    for sequence in REPLICA_SEQUENCES:
        print(f"\n=== Replica/{sequence} ===", flush=True)
        sequence_dir = require_directory(
            args.data_root / "Replica" / sequence, "Replica sequence"
        )
        gt_trajectory = require_file(sequence_dir / "traj.txt", "GT trajectory")
        gt_mesh = require_file(args.gt_mesh_root / f"{sequence}.ply", "GT mesh")
        images = replica_images(sequence_dir)
        jobs_for_sequence = {
            method: by_pair[(sequence, method)] for method in METHODS
        }
        per_method_metrics = {
            method: metric_by_frame(job, images)
            for method, job in jobs_for_sequence.items()
        }
        evaluated_frames = {
            method: sorted(values)
            for method, values in per_method_metrics.items()
        }
        write_json(
            output_root / sequence / "evaluated_frames.json",
            evaluated_frames,
        )
        results[sequence] = {}

        for method in METHODS:
            job = jobs_for_sequence[method]
            method_output = output_root / sequence / method
            if method == "photoslam" and args.photoslam_run_id:
                method_output = method_output / args.photoslam_run_id
            values = per_method_metrics[method]
            frame_ids = evaluated_frames[method]
            lpips_values: dict[int, float]
            if args.skip_lpips:
                lpips_values = {frame: math.nan for frame in frame_ids}
            else:
                lpips_values = lpips_for_frames(
                    job,
                    images,
                    values,
                    frame_ids,
                    args.lpips_device,
                    method_output / "lpips_per_frame.json",
                    lpips_runtime,
                )

            ate_cm = evaluate_ate(
                job,
                gt_trajectory,
                method_output / "tracking",
                args.evo_ape,
            )
            if args.skip_reconstruction:
                geometry = {
                    key: math.nan
                    for key in (
                        "mean precision",
                        "mean recall",
                        "precision",
                        "recall",
                        "f-score",
                    )
                }
            else:
                geometry = evaluate_geometry(
                    job,
                    gt_mesh,
                    gt_trajectory,
                    method_output / "geometry",
                    args.distance_threshold_m,
                    args.hislam_python,
                    args.force_reconstruction,
                )

            row = {
                "evaluated_frames": len(frame_ids),
                "ate_rmse_cm": ate_cm,
                "accuracy_cm": 100.0 * geometry["mean precision"],
                "completeness_cm": 100.0 * geometry["mean recall"],
                "completion_ratio_percent": 100.0 * geometry["recall"],
                "precision_5cm": geometry["precision"],
                "recall_5cm": geometry["recall"],
                "fscore_5cm": geometry["f-score"],
                "psnr": mean(values[frame]["psnr"] for frame in frame_ids),
                "ssim": mean(values[frame]["ssim"] for frame in frame_ids),
                "lpips": mean(lpips_values[frame] for frame in frame_ids),
                "shutdown_dir": str(job.shutdown_dir),
                "raw_mesh": str(raw_mesh(job)),
                "gt_mesh": str(gt_mesh),
            }
            results[sequence][method] = row
            write_json(method_output / "metrics.json", row)
            print(
                f"[{sequence}/{method}] ATE={row['ate_rmse_cm']:.3f} cm, "
                f"Acc={row['accuracy_cm']:.3f} cm, "
                f"Comp={row['completeness_cm']:.3f} cm, "
                f"CompRat={row['completion_ratio_percent']:.2f}%, "
                f"PSNR={row['psnr']:.3f}, SSIM={row['ssim']:.4f}, "
                f"LPIPS={row['lpips']:.4f}",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "photoslam_run_id": args.photoslam_run_id or args.run_id,
        "protocol": {
            "tracking": "evo_ape TUM ATE RMSE with Sim(3) alignment",
            "geometry": (
                "trajectory Sim(3) through mesh_eval, then HI-SLAM2 rigid "
                "ICP and 3D reconstruction evaluation"
            ),
            "distance_threshold_m": args.distance_threshold_m,
            "gt_mesh_root": str(args.gt_mesh_root.resolve()),
            "appearance": (
                "saved keyframe PSNR/SSIM and AlexNet LPIPS over every "
                "keyframe rendered by each completed mapping run"
            ),
            "averaging": "arithmetic mean of the eight per-sequence metrics",
        },
        "sequences": results,
        "average": aggregate(results),
    }
    summary_path = output_root / "replica_summary.json"
    write_json(summary_path, summary)
    print(f"\nSaved Replica summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
