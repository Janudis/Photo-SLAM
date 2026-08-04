#!/usr/bin/env python3
"""Export a MonoGS run as a rendered-depth TSDF surface mesh.

Run this file with the MonoGS conda environment. It uses MonoGS's own Gaussian
renderer and HI-SLAM2's TSDF integration script. The output mesh remains in the
native MonoGS frame; matched estimated and GT keyframe trajectories are saved
for the benchmark's trajectory-based Sim(3) alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from evo.core import trajectory
from evo.core.trajectory import PosePath3D
from munch import munchify
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
MONOGS_ROOT = REPO_ROOT / "third_party/MonoGS"
HI_SLAM2_ROOT = REPO_ROOT / "third_party/HI-SLAM2"
HI_SLAM2_TSDF = HI_SLAM2_ROOT / "tsdf_integrate.py"
HI_SLAM2_PYTHON = (
    Path.home() / "miniconda3/envs/hislam2/bin/python"
)

VOXEL_SIZE_M = 0.03
MIN_WEIGHT = 1.0
TRUNCATION_VOXELS = 8.0
DEPTH_MAX_M = 40.0
ALPHA_THRESHOLD = 0.5
DEPTH_SCALE = 6553.5

sys.path.insert(0, str(MONOGS_ROOT))

from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_splatting.utils.graphics_utils import (  # noqa: E402
    getProjectionMatrix2,
)
from utils.camera_utils import Camera  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a completed MonoGS map at its estimated keyframes and "
            "extract a HI-SLAM2-style TSDF surface mesh."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="completed MonoGS timestamped run directory",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def load_mapping(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a mapping: {path}")
    return value


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_file(path, "MonoGS final trajectory")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        frame_ids = np.asarray(value["trj_id"], dtype=np.int64)
        estimated = np.asarray(value["trj_est"], dtype=np.float64)
        ground_truth = np.asarray(value["trj_gt"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid MonoGS trajectory: {path}") from error

    count = frame_ids.shape[0]
    if count == 0:
        raise RuntimeError(f"MonoGS trajectory is empty: {path}")
    if estimated.shape != (count, 4, 4) or ground_truth.shape != (count, 4, 4):
        raise RuntimeError(f"Unexpected MonoGS trajectory dimensions: {path}")
    if len(set(frame_ids.tolist())) != count:
        raise RuntimeError(f"Duplicate frame IDs in MonoGS trajectory: {path}")
    if not (
        np.isfinite(estimated).all()
        and np.isfinite(ground_truth).all()
        and np.isfinite(frame_ids).all()
    ):
        raise RuntimeError(f"Non-finite MonoGS trajectory data: {path}")
    return frame_ids, estimated, ground_truth


def pose_vector(timestamp: float, c2w: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    return np.concatenate(
        (
            np.asarray([timestamp], dtype=np.float64),
            c2w[:3, 3],
            quaternion,
        )
    )


def write_trajectories(
    output_dir: Path,
    frame_ids: np.ndarray,
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[Path, Path, Path]:
    integration_trajectory = output_dir / "traj_full.txt"
    recon_trajectory = output_dir / "CameraTrajectory_TUM.txt"
    gt_trajectory = output_dir / "gt_trajectory_row_major.txt"
    frame_map = output_dir / "frame_id_map.txt"

    integration_rows = np.stack(
        [pose_vector(index, pose) for index, pose in enumerate(estimated)]
    )
    recon_rows = np.stack(
        [
            pose_vector(float(frame_id), pose)
            for frame_id, pose in zip(frame_ids, estimated)
        ]
    )
    np.savetxt(integration_trajectory, integration_rows, fmt="%.10f")
    np.savetxt(recon_trajectory, recon_rows, fmt="%.10f")
    np.savetxt(
        gt_trajectory,
        ground_truth.reshape(ground_truth.shape[0], 16),
        fmt="%.10f",
    )
    np.savetxt(
        frame_map,
        np.column_stack((np.arange(frame_ids.shape[0]), frame_ids)),
        fmt="%d",
        header="integration_index dataset_frame_id",
    )
    return integration_trajectory, recon_trajectory, gt_trajectory


def resolve_dataset_path(config: dict[str, Any]) -> None:
    raw_path = Path(str(config["Dataset"]["dataset_path"])).expanduser()
    if not raw_path.is_absolute():
        raw_path = MONOGS_ROOT / raw_path
    config["Dataset"]["dataset_path"] = str(raw_path.resolve())


def render_keyframes(
    run_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    frame_ids: np.ndarray,
    estimated: np.ndarray,
    scale: float,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MonoGS reconstruction export requires CUDA")

    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])
    sh_degree = 3 if config["Training"]["spherical_harmonics"] else 0
    gaussians = GaussianModel(sh_degree, config=config)
    gaussian_path = require_file(
        run_dir / "point_cloud/final/point_cloud.ply",
        "MonoGS final Gaussian map",
    )
    gaussians.load_ply(str(gaussian_path))
    dataset = load_dataset(
        model_params,
        model_params.source_path,
        config=config,
    )

    if int(frame_ids.min()) < 0 or int(frame_ids.max()) >= len(dataset):
        raise RuntimeError(
            "MonoGS trajectory contains frame IDs outside the configured dataset"
        )

    projection = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)
    projection = projection.to(device="cuda:0")
    background = torch.zeros(3, dtype=torch.float32, device="cuda:0")

    depth_dir = output_dir / "renders/depth_after_opt"
    image_dir = output_dir / "renders/image_after_opt"
    depth_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    depth_max_native = DEPTH_MAX_M / scale
    max_encodable_depth = 65535.0 / DEPTH_SCALE
    for index, (frame_id, c2w) in enumerate(zip(frame_ids, estimated)):
        camera = Camera.init_from_dataset(dataset, int(frame_id), projection)
        w2c = np.linalg.inv(c2w)
        camera.update_RT(
            torch.as_tensor(
                w2c[:3, :3],
                dtype=torch.float32,
                device="cuda:0",
            ),
            torch.as_tensor(
                w2c[:3, 3],
                dtype=torch.float32,
                device="cuda:0",
            ),
        )

        with torch.no_grad():
            package = render(
                camera,
                gaussians,
                pipeline_params,
                background,
            )

        depth = package["depth"].detach().squeeze().cpu().numpy()
        opacity = package["opacity"].detach().squeeze().cpu().numpy()
        color = (
            package["render"]
            .detach()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        valid = (
            np.isfinite(depth)
            & np.isfinite(opacity)
            & (depth > 0.0)
            & (depth <= depth_max_native)
            & (depth <= max_encodable_depth)
            & (opacity >= ALPHA_THRESHOLD)
        )
        depth_output = np.zeros(depth.shape, dtype=np.uint16)
        depth_output[valid] = np.rint(
            depth[valid] * DEPTH_SCALE
        ).astype(np.uint16)
        color_output = cv2.cvtColor(
            np.rint(color * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

        filename = f"{index:06d}.png"
        if not cv2.imwrite(str(depth_dir / filename), depth_output):
            raise RuntimeError(f"Failed to write MonoGS depth {filename}")
        if not cv2.imwrite(str(image_dir / filename), color_output):
            raise RuntimeError(f"Failed to write MonoGS color {filename}")

        del camera, package
        if (index + 1) % 25 == 0 or index + 1 == frame_ids.shape[0]:
            print(
                f"[MonoGS mesh] rendered {index + 1}/{frame_ids.shape[0]} "
                "keyframes",
                flush=True,
            )


def run_tsdf_integration(
    output_dir: Path,
    integration_trajectory: Path,
    scale: float,
) -> Path:
    require_file(HI_SLAM2_TSDF, "HI-SLAM2 TSDF integration script")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python interpreter")
    voxel_size_native = VOXEL_SIZE_M / scale
    depth_max_native = DEPTH_MAX_M / scale
    command = [
        str(HI_SLAM2_PYTHON),
        str(HI_SLAM2_TSDF),
        "--result",
        str(output_dir),
        "--voxel_size",
        f"{voxel_size_native:.10g}",
        "--depth_scale",
        f"{DEPTH_SCALE:.10g}",
        "--depth_max",
        f"{depth_max_native:.10g}",
        "--weight",
        f"{MIN_WEIGHT:g}",
        "--iteration",
        "after_opt",
        "--traj",
        str(integration_trajectory),
    ]
    print("[MonoGS mesh] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=HI_SLAM2_ROOT, check=True)

    generated_mesh = require_file(
        output_dir / f"tsdf_mesh_w{MIN_WEIGHT:.1f}.ply",
        "HI-SLAM2 TSDF mesh",
    )
    final_mesh = output_dir / "monogs_surface_mesh.ply"
    shutil.copy2(generated_mesh, final_mesh)
    return final_mesh


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"MonoGS run directory does not exist: {run_dir}")

    config = load_mapping(run_dir / "config.yml", "MonoGS saved config")
    if config.get("Dataset", {}).get("type") != "replica":
        raise RuntimeError("This exporter currently supports MonoGS Replica runs")
    if config["Dataset"].get("sensor_type") != "monocular":
        raise RuntimeError("Expected a monocular MonoGS run")
    resolve_dataset_path(config)

    frame_ids, estimated, ground_truth = load_trajectory(
        run_dir / "plot/trj_final.json"
    )
    estimated_path = PosePath3D(poses_se3=list(estimated))
    ground_truth_path = PosePath3D(poses_se3=list(ground_truth))
    _, rotation, translation, scale = trajectory.align_trajectory(
        estimated_path,
        ground_truth_path,
        correct_scale=True,
        return_parameters=True,
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Invalid MonoGS trajectory scale: {scale}")

    output_dir = run_dir / "reconstruction_eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    integration_trajectory, recon_trajectory, gt_trajectory = (
        write_trajectories(
            output_dir,
            frame_ids,
            estimated,
            ground_truth,
        )
    )
    np.save(
        output_dir / "intrinsics.npy",
        np.asarray(
            [
                config["Dataset"]["Calibration"]["fx"],
                config["Dataset"]["Calibration"]["fy"],
                config["Dataset"]["Calibration"]["cx"],
                config["Dataset"]["Calibration"]["cy"],
            ],
            dtype=np.float64,
        ),
    )

    render_keyframes(
        run_dir,
        output_dir,
        config,
        frame_ids,
        estimated,
        float(scale),
    )
    mesh = run_tsdf_integration(
        output_dir,
        integration_trajectory,
        float(scale),
    )

    metadata = {
        "method": "MonoGS",
        "map_stage": "after 26000-iteration color refinement",
        "source_run": str(run_dir),
        "source_gaussians": str(
            run_dir / "point_cloud/final/point_cloud.ply"
        ),
        "surface_mesh": str(mesh),
        "recon_trajectory": str(recon_trajectory),
        "matched_gt_trajectory": str(gt_trajectory),
        "evaluated_keyframes": int(frame_ids.shape[0]),
        "input_resolution": [
            int(config["Dataset"]["Calibration"]["width"]),
            int(config["Dataset"]["Calibration"]["height"]),
        ],
        "sim3": {
            "scale": float(scale),
            "rotation": np.asarray(rotation).tolist(),
            "translation": np.asarray(translation).tolist(),
        },
        "tsdf": {
            "metric_voxel_size_m": VOXEL_SIZE_M,
            "native_voxel_size": VOXEL_SIZE_M / float(scale),
            "minimum_weight": MIN_WEIGHT,
            "truncation_voxels": TRUNCATION_VOXELS,
            "metric_depth_max_m": DEPTH_MAX_M,
            "alpha_threshold": ALPHA_THRESHOLD,
            "depth_scale": DEPTH_SCALE,
            "integration": str(HI_SLAM2_TSDF),
        },
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[MonoGS mesh] saved: {mesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
