#!/usr/bin/env python3
"""Run TANDEM on Replica office0 and materialize fair evaluation inputs.

Run from the Photo-SLAM root:
    python3 evaluation/run_tandem_replica.py --preset runtime
    python3 evaluation/run_tandem_replica.py --preset dataset
    python3 evaluation/evaluate_monocular_densification_ablation.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
TANDEM_REPOSITORY = REPO_ROOT / "third_party/tandem"
TANDEM_ROOT = REPO_ROOT / "third_party/tandem/tandem"
TANDEM_BUILD = TANDEM_ROOT / "build"
TANDEM_BINARY = TANDEM_BUILD / "bin/tandem_dataset"
TANDEM_MODEL = TANDEM_ROOT / "exported/tandem_512x320"
SOURCE_SCENE = REPO_ROOT / "scripts/data/Replica/office0"
SOURCE_IMAGES = SOURCE_SCENE / "results"
SOURCE_GT_TRAJECTORY = SOURCE_SCENE / "traj.txt"
TANDEM_SCENE = TANDEM_ROOT / "data/Replica/office0"
RESULT_DIRS = {
    "runtime": REPO_ROOT / "results/tandem/replica/office0",
    "dataset": REPO_ROOT / "results/tandem/replica/office0_dataset",
}
HI_SLAM2_PYTHON = Path.home() / "miniconda3/envs/hislam2/bin/python"
INSTRUMENTATION_PATCH = REPO_ROOT / "evaluation/tandem_runtime_metrics.patch"
FRAME_COUNT = 2000
FRAME_PERIOD_SECONDS = 1.0 / 30.0
GPU_POLL_SECONDS = 0.25


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def prepare_scene() -> None:
    TANDEM_SCENE.mkdir(parents=True, exist_ok=True)
    image_dir = TANDEM_SCENE / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in range(FRAME_COUNT):
        name = f"frame{frame_id:06d}.jpg"
        source = SOURCE_IMAGES / name
        require_file(source, "Replica color image")
        destination = image_dir / name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(
                    f"Existing TANDEM image link has the wrong target: {destination}"
                )
        elif destination.exists():
            if destination.resolve() != source.resolve():
                raise RuntimeError(
                    f"Refusing to replace existing TANDEM image: {destination}"
                )
        else:
            destination.symlink_to(source.resolve())

    camera_text = (
        "Pinhole 600 600 599.5 339.5 0\n"
        "1200 680\n"
        "crop\n"
        "512 320\n"
    )
    (TANDEM_SCENE / "camera.txt").write_text(camera_text, encoding="utf-8")
    (TANDEM_SCENE / "rgb.txt").write_text(
        "".join(
            f"{frame_id} {frame_id * FRAME_PERIOD_SECONDS:.9f} 1.0\n"
            for frame_id in range(FRAME_COUNT)
        ),
        encoding="utf-8",
    )


def ensure_runtime_instrumentation() -> None:
    main_source = TANDEM_ROOT / "src/main_tandem_pangolin.cpp"
    timer_source = TANDEM_ROOT / "src/util/Timer.h"
    require_file(main_source, "TANDEM dataset entry point")
    require_file(timer_source, "TANDEM timer")
    require_file(INSTRUMENTATION_PATCH, "TANDEM runtime instrumentation patch")
    main_text = main_source.read_text(encoding="utf-8")
    timer_text = timer_source.read_text(encoding="utf-8")
    if "saveRuntimeMetrics(" in main_text and "sum_timing_ms(" in timer_text:
        return
    if "saveRuntimeMetrics(" in main_text or "sum_timing_ms(" in timer_text:
        raise RuntimeError(
            "TANDEM runtime instrumentation is only partially applied; "
            "restore those two files before running the evaluation"
        )
    subprocess.run(
        ["git", "apply", "--check", str(INSTRUMENTATION_PATCH)],
        cwd=TANDEM_REPOSITORY,
        check=True,
    )
    subprocess.run(
        ["git", "apply", str(INSTRUMENTATION_PATCH)],
        cwd=TANDEM_REPOSITORY,
        check=True,
    )


def stream_output(pipe: TextIO, log_file: TextIO) -> None:
    for line in pipe:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def process_gpu_memory_mb(pid: int, nvidia_smi: str) -> float | None:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            process_id = int(fields[0])
            used_mb = float(fields[1])
        except ValueError:
            continue
        if process_id == pid and math.isfinite(used_mb):
            return used_mb
    return None


def run_tandem(preset: str, result_dir: Path) -> float:
    result_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_instrumentation()
    subprocess.run(
        [
            "cmake",
            "--build",
            str(TANDEM_BUILD),
            "--target",
            "tandem_dataset",
            "-j8",
        ],
        cwd=TANDEM_ROOT,
        check=True,
    )
    require_file(TANDEM_BINARY, "TANDEM executable")
    require_file(TANDEM_MODEL / "model.pt", "TANDEM MVS model")
    require_file(TANDEM_MODEL / "sample_inputs.pt", "TANDEM sample inputs")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for process-level VRAM measurement")
    nvidia_smi_check = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if nvidia_smi_check.returncode != 0:
        raise RuntimeError(
            "nvidia-smi cannot access the GPU: " + nvidia_smi_check.stderr.strip()
        )

    command = [
        str(TANDEM_BINARY),
        f"preset={preset}",
        f"result_folder={result_dir}",
        f"files={TANDEM_SCENE / 'images'}",
        f"calib={TANDEM_SCENE / 'camera.txt'}",
        f"mvsnet_folder={TANDEM_MODEL}",
        "exit_when_done=1",
        "mode=2",
        "dr_timing=1",
        "mesh_extraction_freq=0",
        "nogui=1",
        "nolog=1",
    ]
    (result_dir / "command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    peak_gpu_mb = 0.0
    with (result_dir / "out.txt").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=TANDEM_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_thread = threading.Thread(
            target=stream_output,
            args=(process.stdout, log_file),
            daemon=True,
        )
        output_thread.start()
        while process.poll() is None:
            used_mb = process_gpu_memory_mb(process.pid, nvidia_smi)
            if used_mb is not None:
                peak_gpu_mb = max(peak_gpu_mb, used_mb)
            time.sleep(GPU_POLL_SECONDS)
        output_thread.join()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"TANDEM failed with exit code {return_code}")
    if peak_gpu_mb <= 0.0:
        raise RuntimeError("No TANDEM process-level GPU memory sample was recorded")
    return peak_gpu_mb


def convert_mesh(result_dir: Path) -> Path:
    source = result_dir / "mesh.obj"
    destination = result_dir / "mesh.ply"
    require_file(source, "TANDEM TSDF mesh")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python")
    conversion = """
import sys
import trimesh

mesh = trimesh.load(sys.argv[1], process=False)
if isinstance(mesh, trimesh.Scene):
    if not mesh.geometry:
        raise RuntimeError("TANDEM OBJ contains no geometry")
    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
    raise RuntimeError("TANDEM OBJ contains no triangle mesh")
mesh.export(sys.argv[2], file_type="ply", encoding="binary_little_endian")
"""
    subprocess.run(
        [str(HI_SLAM2_PYTHON), "-c", conversion, str(source), str(destination)],
        check=True,
    )
    require_file(destination, "converted TANDEM PLY mesh")
    return destination.resolve()


def parse_gt_trajectory() -> list[list[float]]:
    require_file(SOURCE_GT_TRAJECTORY, "Replica GT trajectory")
    matrices: list[list[float]] = []
    for line_number, line in enumerate(
        SOURCE_GT_TRAJECTORY.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            values = [float(token) for token in line.split()]
        except ValueError as error:
            raise RuntimeError(
                f"Invalid GT trajectory value at line {line_number}"
            ) from error
        if len(values) != 16 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Expected 16 finite GT values at line {line_number}"
            )
        matrices.append(values)
    if len(matrices) != FRAME_COUNT:
        raise RuntimeError(
            f"Expected {FRAME_COUNT} Replica GT poses; found {len(matrices)}"
        )
    return matrices


def materialize_matched_trajectories(result_dir: Path) -> tuple[Path, Path]:
    result_path = result_dir / "result.txt"
    require_file(result_path, "TANDEM valid-pose trajectory")
    pose_lines = [
        line.strip()
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(pose_lines) < 3:
        raise RuntimeError("TANDEM produced fewer than three valid poses")

    rgb_rows = []
    for line in (TANDEM_SCENE / "rgb.txt").read_text(encoding="utf-8").splitlines():
        frame_token, timestamp_token, _ = line.split()
        rgb_rows.append((float(timestamp_token), int(frame_token)))
    rgb_timestamps = [row[0] for row in rgb_rows]
    gt_matrices = parse_gt_trajectory()
    matched_gt: list[list[float]] = []
    previous_frame_id = -1
    for line_number, line in enumerate(pose_lines, start=1):
        tokens = line.split()
        if len(tokens) != 8:
            raise RuntimeError(
                f"Expected TUM-format TANDEM pose at {result_path}:{line_number}"
            )
        timestamp = float(tokens[0])
        if not all(math.isfinite(float(token)) for token in tokens):
            raise RuntimeError(f"Non-finite TANDEM pose at line {line_number}")
        insertion = bisect.bisect_left(rgb_timestamps, timestamp)
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(rgb_timestamps)
        ]
        nearest = min(candidates, key=lambda index: abs(rgb_timestamps[index] - timestamp))
        if abs(rgb_timestamps[nearest] - timestamp) > 1.0e-6:
            raise RuntimeError(
                f"Cannot match TANDEM timestamp {timestamp:.9f} to Replica input"
            )
        frame_id = rgb_rows[nearest][1]
        if frame_id <= previous_frame_id:
            raise RuntimeError("TANDEM valid poses are not strictly frame-ordered")
        previous_frame_id = frame_id
        matched_gt.append(gt_matrices[frame_id])

    recon_path = result_dir / "CameraTrajectory_TUM.txt"
    recon_path.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    matched_gt_path = result_dir / "gt_trajectory_matched.txt"
    matched_gt_path.write_text(
        "\n".join(" ".join(f"{value:.17g}" for value in matrix) for matrix in matched_gt)
        + "\n",
        encoding="utf-8",
    )
    return recon_path.resolve(), matched_gt_path.resolve()


def finalize_metrics(
    preset: str,
    result_dir: Path,
    peak_process_gpu_mb: float,
    mesh_path: Path,
    recon_path: Path,
    matched_gt_path: Path,
) -> None:
    metrics_path = result_dir / "runtime_metrics.json"
    require_file(
        metrics_path,
        "instrumented TANDEM runtime metrics; rebuild TANDEM with this checkout",
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics.get("frames", 0)) != FRAME_COUNT:
        raise RuntimeError(
            f"TANDEM processed {metrics.get('frames')} frames, expected {FRAME_COUNT}"
        )
    metrics.update(
        {
            "preset": preset,
            "map_path": str(mesh_path),
            "map_size_mb": mesh_path.stat().st_size / (1024.0 * 1024.0),
            "gpu_process_peak_used_mb": peak_process_gpu_mb,
            "gpu_process_memory_scope": (
                "nvidia-smi per-process used memory; includes LibTorch and custom CUDA allocations"
            ),
            "recon_trajectory": str(recon_path),
            "alignment_gt_trajectory": str(matched_gt_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TANDEM Replica office0 with a quality or runtime preset."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(RESULT_DIRS),
        default="runtime",
        help="dataset is used for reconstruction quality; runtime is used for FPS/VRAM",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = RESULT_DIRS[args.preset]
    prepare_scene()
    peak_process_gpu_mb = run_tandem(args.preset, result_dir)
    mesh_path = convert_mesh(result_dir)
    recon_path, matched_gt_path = materialize_matched_trajectories(result_dir)
    finalize_metrics(
        args.preset,
        result_dir,
        peak_process_gpu_mb,
        mesh_path,
        recon_path,
        matched_gt_path,
    )
    print(f"\nTANDEM Replica outputs: {result_dir}")
    print("Run the common evaluator with:")
    print("  python3 evaluation/evaluate_monocular_densification_ablation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
