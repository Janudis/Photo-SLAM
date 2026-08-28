#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPLICA_SEQUENCES = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)

SCANNET_SEQUENCES = (
    "scene0000_00",
    "scene0059_00",
    "scene0106_00",
    "scene0169_00",
    "scene0181_00",
    "scene0207_00",
)

SOURCE_COMMITS = {
    "monogs": "6c9254c319d8bff5caeef65259e6bb0941a9b9f6",
    "hislam2": "76c833c7d8ed474f0f3ba18056c1803e032a537f",
    "tandem": "f8816c7d9a92b29e84e3d9055c2d3e28056e4a37",
}

SOURCE_ROOTS = {
    "monogs": Path("/opt/MonoGS"),
    "hislam2": Path("/opt/HI-SLAM2"),
    "tandem": Path("/opt/tandem/tandem"),
}


class BaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = ()


@dataclass(frozen=True)
class PreparedInput:
    root: Path
    images: Path
    depths: Path | None
    calibration: CameraCalibration
    calibration_file: Path
    frame_count: int
    use_undistortion: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_run_lock(path: Path):
    handle = path.open("w", encoding="ascii")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise BaselineError(
            f"Benchmark run is already active: {path.parent.name}"
        ) from error
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise BaselineError(f"Missing {description}: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise BaselineError(f"Missing {description}: {path}")
    return path


def replace_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def symlink_files(files: Iterable[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in files:
        target = destination / source.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
        count += 1
    return count


def symlink_numbered(
    files: Iterable[Path],
    destination: Path,
    prefix: str,
    suffix: str,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, source in enumerate(files):
        target = destination / f"{prefix}{index:06d}{suffix}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
        count += 1
    return count


def natural_path_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def symlink_ordered(files: Iterable[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, source in enumerate(files):
        target = destination / f"{index:06d}{source.suffix.lower()}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
        count += 1
    return count


def replica_dir(data_root: Path, sequence: str) -> Path:
    candidates = (
        data_root / "Replica" / sequence,
        data_root / "replica" / sequence,
    )
    for candidate in candidates:
        if (candidate / "results").is_dir() and (candidate / "traj.txt").is_file():
            return candidate
    raise BaselineError(f"Cannot locate Replica sequence {sequence} below {data_root}")


def scannet_dir(data_root: Path, sequence: str) -> Path:
    candidates = (
        data_root / "ScanNet" / "scans" / sequence,
        data_root / "ScanNet" / sequence,
        data_root / "scannet" / "scans" / sequence,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise BaselineError(f"Cannot locate ScanNet sequence {sequence} below {data_root}")


def tum_dir(data_root: Path, sequence: str) -> Path:
    candidates = (
        data_root / "TUM" / sequence,
        data_root / sequence,
    )
    for candidate in candidates:
        if (candidate / "rgb.txt").is_file():
            return candidate
    raise BaselineError(f"Cannot locate TUM sequence {sequence} below {data_root}")


def parse_tum_list(path: Path) -> list[Path]:
    values: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            values.append(path.parent / fields[1])
    return values


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise BaselineError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    return width, height


def load_scannet_calibration(scene: Path, image: Path) -> CameraCalibration:
    import numpy as np

    width, height = image_size(image)
    calib = scene / "calib.txt"
    if calib.is_file():
        values = np.loadtxt(calib).reshape(-1)
        if values.size >= 4:
            return CameraCalibration(
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                width,
                height,
            )

    intrinsic = scene / "intrinsic" / "intrinsic_color.txt"
    matrix = np.loadtxt(require_file(intrinsic, "ScanNet color intrinsics"))
    return CameraCalibration(
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        width,
        height,
    )


def tum_calibration(sequence: str) -> CameraCalibration:
    if sequence == "rgbd_dataset_freiburg1_desk":
        return CameraCalibration(
            517.306408,
            516.469215,
            318.643040,
            255.313989,
            640,
            480,
            (0.262383, -0.953104, -0.005358, 0.002628, 1.163314),
        )
    raise BaselineError(
        f"No reviewed calibration is registered for TUM sequence {sequence}"
    )


def write_calibration_values(path: Path, calibration: CameraCalibration) -> None:
    values = [
        calibration.fx,
        calibration.fy,
        calibration.cx,
        calibration.cy,
        *calibration.distortion,
    ]
    path.write_text(" ".join(f"{value:.12g}" for value in values) + "\n", encoding="ascii")


def write_dso_calibration(path: Path, calibration: CameraCalibration) -> None:
    model = "Pinhole"
    distortion = "0"
    if calibration.distortion:
        model = "RadTan"
        distortion = " ".join(f"{value:.12g}" for value in calibration.distortion[:4])
    line = " ".join(
        (
            model,
            f"{calibration.fx / calibration.width:.12g}",
            f"{calibration.fy / calibration.height:.12g}",
            f"{calibration.cx / calibration.width:.12g}",
            f"{calibration.cy / calibration.height:.12g}",
            distortion,
        )
    )
    path.write_text(
        f"{line}\n{calibration.width} {calibration.height}\ncrop\n512 320\n",
        encoding="ascii",
    )


def prepare_replica(data_root: Path, sequence: str, workspace: Path) -> PreparedInput:
    source = replica_dir(data_root, sequence)
    results = source / "results"
    images = sorted(results.glob("frame*.jpg"))
    depths = sorted(results.glob("depth*.png"))
    if not images:
        raise BaselineError(f"No Replica RGB images in {results}")

    colors_dir = workspace / "colors"
    depths_dir = workspace / "depths"
    symlink_files(images, colors_dir)
    symlink_files(depths, depths_dir)
    monogs_results = workspace / "results"
    symlink_files(images, monogs_results)
    symlink_files(depths, monogs_results)
    (workspace / "traj.txt").symlink_to((source / "traj.txt").resolve())

    calibration = CameraCalibration(600.0, 600.0, 599.5, 339.5, 1200, 680)
    calibration_file = workspace / "calib.txt"
    write_calibration_values(calibration_file, calibration)
    return PreparedInput(
        workspace,
        colors_dir,
        depths_dir,
        calibration,
        calibration_file,
        len(images),
    )


def prepare_scannet(data_root: Path, sequence: str, workspace: Path) -> PreparedInput:
    import numpy as np

    scene = scannet_dir(data_root, sequence)
    color_dir = scene / "color"
    if not color_dir.is_dir():
        color_dir = scene / "results"
    images = sorted(
        (*color_dir.glob("*.jpg"), *color_dir.glob("*.png")),
        key=natural_path_key,
    )
    if not images:
        raise BaselineError(f"No ScanNet RGB images in {color_dir}")

    source_depth = scene / "depth"
    depths = (
        sorted(source_depth.glob("*.png"), key=natural_path_key)
        if source_depth.is_dir()
        else []
    )
    colors_dir = workspace / "colors"
    depths_dir = workspace / "depths"
    symlink_ordered(images, colors_dir)
    symlink_ordered(depths, depths_dir)
    monogs_results = workspace / "results"
    symlink_numbered(images, monogs_results, "frame", ".jpg")
    symlink_numbered(depths, monogs_results, "depth", ".png")

    poses_dir = require_dir(scene / "pose", "ScanNet poses")
    pose_paths = sorted(poses_dir.glob("*.txt"), key=natural_path_key)
    trajectory: list[list[float]] = []
    last_valid = np.eye(4, dtype=np.float64)
    invalid_ids: list[int] = []
    for index in range(len(images)):
        pose_path = poses_dir / f"{index}.txt"
        if not pose_path.is_file() and index < len(pose_paths):
            pose_path = pose_paths[index]
        pose = np.loadtxt(require_file(pose_path, "ScanNet pose")).reshape(4, 4)
        if not np.isfinite(pose).all():
            invalid_ids.append(index)
            pose = last_valid.copy()
        else:
            last_valid = pose.copy()
        trajectory.append(pose.reshape(-1).tolist())
    np.savetxt(workspace / "traj.txt", np.asarray(trajectory))
    write_json(workspace / "invalid_gt_pose_frames.json", invalid_ids)

    calibration = load_scannet_calibration(scene, images[0])
    calibration_file = workspace / "calib.txt"
    write_calibration_values(calibration_file, calibration)
    return PreparedInput(
        workspace,
        colors_dir,
        depths_dir if depths else None,
        calibration,
        calibration_file,
        len(images),
    )


def prepare_tum(data_root: Path, sequence: str, workspace: Path) -> PreparedInput:
    source = tum_dir(data_root, sequence)
    images = parse_tum_list(source / "rgb.txt")
    depths = parse_tum_list(source / "depth.txt")
    if not images:
        raise BaselineError(f"No TUM RGB images in {source}")
    colors_dir = workspace / "colors"
    depths_dir = workspace / "depths"
    symlink_files(images, colors_dir)
    symlink_files(depths, depths_dir)
    calibration = tum_calibration(sequence)
    calibration_file = workspace / "calib.txt"
    write_calibration_values(calibration_file, calibration)
    return PreparedInput(
        workspace,
        colors_dir,
        depths_dir if depths else None,
        calibration,
        calibration_file,
        len(images),
        use_undistortion=bool(calibration.distortion),
    )


def prepare_input(
    dataset: str,
    data_root: Path,
    sequence: str,
    workspace: Path,
) -> PreparedInput:
    replace_directory(workspace)
    if dataset == "replica":
        return prepare_replica(data_root, sequence, workspace)
    if dataset == "scannet":
        return prepare_scannet(data_root, sequence, workspace)
    if dataset == "tum":
        return prepare_tum(data_root, sequence, workspace)
    raise BaselineError(f"Unsupported dataset: {dataset}")


def yaml_quote(value: Path | str) -> str:
    return json.dumps(str(value))


def monogs_config(
    dataset: str,
    sequence: str,
    prepared: PreparedInput,
    output: Path,
) -> Path:
    config_path = output / "monogs_config.yaml"
    if dataset == "tum" and sequence == "rgbd_dataset_freiburg1_desk":
        inherited = "/opt/MonoGS/configs/mono/tum/fr1_desk.yaml"
        dataset_path = tum_dir(Path("/datasets"), sequence)
        text = (
            f"inherit_from: {yaml_quote(inherited)}\n"
            "Results:\n"
            f"  save_dir: {yaml_quote(output / 'native_runs')}\n"
            "Dataset:\n"
            f"  dataset_path: {yaml_quote(str(dataset_path) + '/')}\n"
        )
    else:
        c = prepared.calibration
        text = (
            "inherit_from: \"/opt/MonoGS/configs/rgbd/replica/base_config.yaml\"\n"
            "Results:\n"
            f"  save_dir: {yaml_quote(output / 'native_runs')}\n"
            "Dataset:\n"
            "  sensor_type: \"monocular\"\n"
            "  type: \"replica\"\n"
            f"  dataset_path: {yaml_quote(str(prepared.root) + '/')}\n"
            "  Calibration:\n"
            f"    fx: {c.fx}\n"
            f"    fy: {c.fy}\n"
            f"    cx: {c.cx}\n"
            f"    cy: {c.cy}\n"
            f"    width: {c.width}\n"
            f"    height: {c.height}\n"
            f"    depth_scale: {6553.5 if dataset == 'replica' else 1000.0}\n"
            "    distorted: False\n"
        )
    config_path.write_text(text, encoding="utf-8")
    return config_path


def command_for(
    method: str,
    dataset: str,
    sequence: str,
    prepared: PreparedInput,
    output: Path,
    tandem_preset: str,
) -> list[str]:
    if method == "monogs":
        config = monogs_config(dataset, sequence, prepared, output)
        return ["python3", "slam.py", "--config", str(config), "--eval"]

    if method == "hislam2":
        config = {
            "replica": "/opt/HI-SLAM2/config/replica_config.yaml",
            "scannet": "/opt/HI-SLAM2/config/scannet_config.yaml",
            "tum": "/opt/HI-SLAM2/config/owndata_config.yaml",
        }[dataset]
        command = [
            "python3",
            "demo.py",
            "--imagedir",
            str(prepared.images),
            "--calib",
            str(prepared.calibration_file),
            "--config",
            config,
            "--output",
            str(output / "native"),
        ]
        if dataset == "replica" and prepared.depths is not None:
            command.extend(("--gtdepthdir", str(prepared.depths)))
        if prepared.use_undistortion:
            command.append("--undistort")
        return command

    if method == "tandem":
        dso_calib = prepared.root / "camera_dso.txt"
        write_dso_calibration(dso_calib, prepared.calibration)
        mode = "2" if dataset == "replica" else "1"
        native = output / f"native_{tandem_preset}"
        native.mkdir(parents=True, exist_ok=True)
        return [
            "/opt/tandem/tandem/build/bin/tandem_dataset",
            f"preset={tandem_preset}",
            f"result_folder={native}/",
            f"files={prepared.images}",
            f"calib={dso_calib}",
            "mvsnet_folder=/opt/tandem/tandem/exported/tandem_512x320",
            f"mode={mode}",
            "nogui=1",
            "nolog=1",
            "quiet=1",
            "dr_timing=1",
            "exit_when_done=1",
            "mesh_extraction_freq=0",
        ]

    raise BaselineError(f"Unsupported method: {method}")


def gpu_memory_used_mb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def terminate_process_group(process: subprocess.Popen, timeout: float = 10.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait()


def run_monitored(command: list[str], cwd: Path, log_path: Path) -> dict[str, float | int | None]:
    baseline = gpu_memory_used_mb()
    peak = baseline
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "WANDB_MODE": "disabled",
            },
            start_new_session=True,
        )
        next_report = start + 10.0
        next_log_check = start + 1.0
        log_offset = 0
        native_traceback = False
        try:
            while process.poll() is None:
                used = gpu_memory_used_mb()
                if used is not None and (peak is None or used > peak):
                    peak = used
                now = time.perf_counter()
                if now >= next_report:
                    gpu_text = "unavailable" if used is None else f"{used:.0f} MiB"
                    print(
                        f"[running] elapsed={now - start:.0f}s "
                        f"GPU={gpu_text} log={log_path}",
                        flush=True,
                    )
                    next_report = now + 30.0
                if now >= next_log_check:
                    log.flush()
                    with log_path.open("rb") as reader:
                        reader.seek(log_offset)
                        output = reader.read()
                        log_offset = reader.tell()
                    if b"Traceback (most recent call last):" in output:
                        native_traceback = True
                        print(
                            "[failed] Native traceback detected; terminating "
                            f"process group. See {log_path}",
                            file=sys.stderr,
                            flush=True,
                        )
                        terminate_process_group(process)
                        break
                    next_log_check = now + 1.0
                time.sleep(0.2)
        except BaseException:
            terminate_process_group(process)
            raise
        return_code = process.wait()
        if native_traceback and return_code == 0:
            return_code = 1
    elapsed = time.perf_counter() - start
    return {
        "return_code": return_code,
        "wall_seconds": elapsed,
        "whole_device_gpu_start_mb": baseline,
        "whole_device_gpu_peak_mb": peak,
        "whole_device_gpu_delta_mb": (
            peak - baseline if peak is not None and baseline is not None else None
        ),
    }


def latest_monogs_run(output: Path) -> Path:
    maps = sorted(output.glob("native_runs/**/point_cloud/final/point_cloud.ply"))
    if not maps:
        raise BaselineError(
            f"MonoGS did not save its final Gaussian map below {output / 'native_runs'}"
        )
    return maps[-1].parents[2]


def find_outputs(method: str, output: Path, tandem_preset: str) -> dict[str, object]:
    values: dict[str, object] = {}
    if method == "monogs":
        maps = sorted(output.glob("native_runs/**/point_cloud/final/point_cloud.ply"))
        metrics = sorted(output.glob("native_runs/**/psnr/after_opt/final_result.json"))
        if maps:
            values["map_path"] = str(maps[-1])
            values["map_size_mb"] = maps[-1].stat().st_size / (1024.0 * 1024.0)
            values["native_output"] = str(maps[-1].parents[2])
        if metrics:
            values["appearance_metrics"] = str(metrics[-1])
        mesh = latest_monogs_run(output) / "reconstruction_eval/monogs_surface_mesh.ply"
        if mesh.is_file():
            values["geometry_meshes"] = [str(mesh)]
            values["reconstruction_metadata"] = str(
                mesh.parent / "export_metadata.json"
            )
    elif method == "hislam2":
        native = output / "native"
        values["native_output"] = str(native)
        for candidate in (
            native / "3dgs_final.ply",
            native / "point_cloud" / "final" / "point_cloud.ply",
        ):
            if candidate.is_file():
                values["map_path"] = str(candidate)
                values["map_size_mb"] = candidate.stat().st_size / (1024.0 * 1024.0)
                break
        metrics = native / "psnr" / "after_opt" / "final_result.json"
        if metrics.is_file():
            values["appearance_metrics"] = str(metrics)
        trajectory = native / "traj_full.txt"
        if trajectory.is_file():
            values["trajectory"] = str(trajectory)
        meshes = sorted(native.glob("tsdf_mesh_w*.ply"))
        if meshes:
            values["geometry_meshes"] = [str(path) for path in meshes]
    else:
        native = output / f"native_{tandem_preset}"
        values["native_output"] = str(native)
        mesh = native / "mesh.obj"
        if mesh.is_file():
            values["map_path"] = str(mesh)
            values["map_size_mb"] = mesh.stat().st_size / (1024.0 * 1024.0)
        trajectory = native / "result.txt"
        if trajectory.is_file():
            values["trajectory"] = str(trajectory)
    return values


def postprocess_monogs(dataset: str, output: Path) -> None:
    if dataset == "tum":
        return
    native_run = latest_monogs_run(output)
    command = [
        "python3",
        "/opt/photoslam-benchmark/export_monogs_reconstruction.py",
        "--run-dir",
        str(native_run),
    ]
    with (output / "reconstruction_export.log").open(
        "w", encoding="utf-8"
    ) as log:
        completed = subprocess.run(
            command,
            cwd=SOURCE_ROOTS["monogs"],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise BaselineError(
            "MonoGS reconstruction export exited with code "
            f"{completed.returncode}; see {output / 'reconstruction_export.log'}"
        )


def postprocess_hislam2(dataset: str, output: Path) -> None:
    if dataset == "tum":
        return
    native = output / "native"
    depth_dir = native / "renders" / "depth_after_opt"
    if not depth_dir.is_dir():
        raise BaselineError(f"HI-SLAM2 did not export rendered depths: {depth_dir}")
    command = [
        "python3",
        "tsdf_integrate.py",
        "--result",
        str(native),
    ]
    if dataset == "replica":
        command.extend(("--voxel_size", "0.006", "--weight", "2"))
    else:
        command.extend(("--voxel_size", "0.015", "--weight", "5", "10"))
    with (output / "tsdf_integration.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=SOURCE_ROOTS["hislam2"],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise BaselineError(
            f"HI-SLAM2 TSDF integration exited with code {completed.returncode}"
        )


def available_tum_sequences(data_root: Path) -> tuple[str, ...]:
    root = data_root / "TUM"
    if not root.is_dir():
        return ()
    return tuple(sorted(path.name for path in root.iterdir() if (path / "rgb.txt").is_file()))


def expand_sequences(dataset: str, values: list[str], data_root: Path) -> tuple[str, ...]:
    if values != ["all"]:
        return tuple(values)
    if dataset == "replica":
        return REPLICA_SEQUENCES
    if dataset == "scannet":
        return SCANNET_SEQUENCES
    sequences = available_tum_sequences(data_root)
    if not sequences:
        raise BaselineError(f"No TUM sequences found below {data_root / 'TUM'}")
    return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pinned external SLAM baselines")
    parser.add_argument("method", choices=("monogs", "hislam2", "tandem"))
    parser.add_argument("dataset", choices=("replica", "tum", "scannet"))
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PHOTOSLAM_DATA_ROOT", "/datasets")))
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("PHOTOSLAM_RESULTS_ROOT", "/results")))
    parser.add_argument("--tandem-preset", choices=("dataset", "runtime"), default="dataset")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = require_dir(SOURCE_ROOTS[args.method], f"{args.method} source")
    sequences = expand_sequences(args.dataset, args.sequences, args.data_root)
    run_root = args.results_root / "paper_benchmark" / args.run_id
    manifest_path = run_root / "run_manifest.json"
    jobs = [
        (sequence, trial)
        for sequence in sequences
        for trial in range(1, args.repetitions + 1)
    ]

    print(f"Method: {args.method} @ {SOURCE_COMMITS[args.method]}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Sequences: {', '.join(sequences)}", flush=True)
    if args.method == "tandem":
        print(f"TANDEM preset: {args.tandem_preset}", flush=True)
    if args.dry_run:
        for sequence, trial in jobs:
            print(f"  {sequence} trial {trial:02d}")
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    _run_lock = acquire_run_lock(run_root / ".run.lock")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "method": args.method,
        "source_commit": SOURCE_COMMITS[args.method],
        "dataset": args.dataset,
        "sequences": list(sequences),
        "repetitions": args.repetitions,
        "tandem_preset": args.tandem_preset if args.method == "tandem" else None,
        "data_root": str(args.data_root),
        "results_root": str(args.results_root),
        "jobs": [],
    }
    write_json(manifest_path, manifest)
    any_failure = False

    for sequence, trial in jobs:
        output = run_root / args.dataset / sequence / args.method / f"trial_{trial:02d}"
        output.mkdir(parents=True, exist_ok=True)
        workspace = Path("/tmp/photoslam-baselines") / args.run_id / args.dataset / sequence / args.method / f"trial_{trial:02d}"
        job: dict[str, object] = {
            "sequence": sequence,
            "trial": trial,
            "status": "running",
            "output": str(output),
            "started_at": utc_now(),
        }
        manifest["jobs"].append(job)
        write_json(manifest_path, manifest)
        print(
            f"\n=== {args.method} | {args.dataset}/{sequence} | trial {trial:02d} ===",
            flush=True,
        )

        try:
            prepared = prepare_input(args.dataset, args.data_root, sequence, workspace)
            command = command_for(
                args.method,
                args.dataset,
                sequence,
                prepared,
                output,
                args.tandem_preset,
            )
            job["command"] = command
            job["frame_count"] = prepared.frame_count
            write_json(
                output / "provenance.json",
                {
                    "method": args.method,
                    "source_commit": SOURCE_COMMITS[args.method],
                    "dataset": args.dataset,
                    "sequence": sequence,
                    "trial": trial,
                    "tandem_preset": args.tandem_preset if args.method == "tandem" else None,
                    "command": command,
                    "frame_count": prepared.frame_count,
                    "calibration": prepared.calibration.__dict__,
                    "created_at": utc_now(),
                },
            )
            print(f"Native log: {output / 'console.log'}", flush=True)
            runtime = run_monitored(command, source_root, output / "console.log")
            runtime["frames"] = prepared.frame_count
            runtime["wall_fps_hz"] = (
                prepared.frame_count / float(runtime["wall_seconds"])
                if float(runtime["wall_seconds"]) > 0
                else 0.0
            )
            write_json(output / "wrapper_runtime_metrics.json", runtime)
            job.update(runtime)
            if runtime["return_code"] != 0:
                raise BaselineError(f"Native process exited with code {runtime['return_code']}")
            if args.method == "monogs":
                postprocess_monogs(args.dataset, output)
            elif args.method == "hislam2":
                postprocess_hislam2(args.dataset, output)
            job.update(find_outputs(args.method, output, args.tandem_preset))
            job["status"] = "complete"
            job["finished_at"] = utc_now()
        except KeyboardInterrupt:
            job["status"] = "interrupted"
            job["finished_at"] = utc_now()
            manifest["status"] = "interrupted"
            manifest["finished_at"] = utc_now()
            write_json(manifest_path, manifest)
            print("\nBenchmark interrupted; native processes stopped.", flush=True)
            return 130
        except Exception as error:
            any_failure = True
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["error"] = str(error)
            write_json(manifest_path, manifest)
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                manifest["status"] = "failed"
                manifest["finished_at"] = utc_now()
                write_json(manifest_path, manifest)
                return 1
        write_json(manifest_path, manifest)

    manifest["status"] = "complete_with_errors" if any_failure else "complete"
    manifest["finished_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"\nBenchmark manifest: {manifest_path}", flush=True)
    return 1 if any_failure else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaselineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
