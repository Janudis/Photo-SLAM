#!/usr/bin/env python3

"""Prepare one Waymo Open Dataset v2 monocular segment for Photo-SLAM.

Vehicle poses are exported only for evaluation and are never fed to the SLAM
system.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "pyarrow is required. Run this script with the pyslam environment: "
        "/home/dimitris/miniconda3/envs/pyslam/bin/python"
    ) from exc


CAMERA_IMAGE = "[CameraImageComponent].image"
CAMERA_EXTRINSIC = "[CameraCalibrationComponent].extrinsic.transform"
VEHICLE_POSE = "[VehiclePoseComponent].world_from_vehicle.transform"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a Waymo v2 FRONT-camera sequence for Photo-SLAM"
    )
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-id", type=int, default=1, help="Waymo FRONT is 1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing images and metadata in the output directory",
    )
    return parser.parse_args()


def find_component(segment_root: Path, component: str) -> Path:
    matches = sorted((segment_root / component).glob("*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {component} parquet under {segment_root}, found {len(matches)}"
        )
    return matches[0]


def matrix4(values: Iterable[float]) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"expected a 4x4 transform, received {matrix.size} values")
    return matrix.reshape(4, 4)


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return an xyzw quaternion from a proper 3x3 rotation matrix."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def camera_axis_transform() -> np.ndarray:
    """Transform an OpenCV optical frame into Waymo's vehicle-like camera frame."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return transform


def write_lines(path: Path, header: List[str], lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for line in header:
            output.write(f"# {line}\n")
        for line in lines:
            output.write(f"{line}\n")


def main() -> None:
    args = parse_args()
    segment_root = args.segment_root.resolve()
    output_root = args.output.resolve()
    image_root = output_root / "rgb"

    if not segment_root.is_dir():
        raise FileNotFoundError(f"Waymo segment does not exist: {segment_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output is not empty: {output_root}; pass --overwrite to replace it"
        )
    image_root.mkdir(parents=True, exist_ok=True)

    image_table = pq.read_table(
        find_component(segment_root, "camera_image"),
        columns=["key.frame_timestamp_micros", "key.camera_name", CAMERA_IMAGE],
    ).to_pydict()
    image_rows = [
        {
            "timestamp_micros": int(timestamp),
            "image": image,
        }
        for timestamp, camera_id, image in zip(
            image_table["key.frame_timestamp_micros"],
            image_table["key.camera_name"],
            image_table[CAMERA_IMAGE],
        )
        if int(camera_id) == args.camera_id
    ]
    image_rows.sort(key=lambda row: row["timestamp_micros"])
    if not image_rows:
        raise RuntimeError(f"camera {args.camera_id} has no images in {segment_root}")
    timestamps = [row["timestamp_micros"] for row in image_rows]
    if len(set(timestamps)) != len(timestamps):
        raise RuntimeError("duplicate FRONT-camera timestamps found")

    calibration_table = pq.read_table(
        find_component(segment_root, "camera_calibration")
    ).to_pydict()
    calibration_rows = [
        index
        for index, camera_id in enumerate(calibration_table["key.camera_name"])
        if int(camera_id) == args.camera_id
    ]
    if len(calibration_rows) != 1:
        raise RuntimeError(
            f"expected one calibration for camera {args.camera_id}, found {len(calibration_rows)}"
        )
    calibration_index = calibration_rows[0]
    calibration: Dict[str, object] = {
        "camera_id": args.camera_id,
        "width": int(calibration_table["[CameraCalibrationComponent].width"][calibration_index]),
        "height": int(calibration_table["[CameraCalibrationComponent].height"][calibration_index]),
        "fx": float(calibration_table["[CameraCalibrationComponent].intrinsic.f_u"][calibration_index]),
        "fy": float(calibration_table["[CameraCalibrationComponent].intrinsic.f_v"][calibration_index]),
        "cx": float(calibration_table["[CameraCalibrationComponent].intrinsic.c_u"][calibration_index]),
        "cy": float(calibration_table["[CameraCalibrationComponent].intrinsic.c_v"][calibration_index]),
        "k1": float(calibration_table["[CameraCalibrationComponent].intrinsic.k1"][calibration_index]),
        "k2": float(calibration_table["[CameraCalibrationComponent].intrinsic.k2"][calibration_index]),
        "p1": float(calibration_table["[CameraCalibrationComponent].intrinsic.p1"][calibration_index]),
        "p2": float(calibration_table["[CameraCalibrationComponent].intrinsic.p2"][calibration_index]),
        "k3": float(calibration_table["[CameraCalibrationComponent].intrinsic.k3"][calibration_index]),
        "vehicle_from_camera": list(calibration_table[CAMERA_EXTRINSIC][calibration_index]),
    }

    pose_table = pq.read_table(
        find_component(segment_root, "vehicle_pose"),
        columns=["key.frame_timestamp_micros", VEHICLE_POSE],
    ).to_pydict()
    world_from_vehicle = {
        int(timestamp): matrix4(transform)
        for timestamp, transform in zip(
            pose_table["key.frame_timestamp_micros"], pose_table[VEHICLE_POSE]
        )
    }

    first_timestamp = timestamps[0]
    vehicle_from_camera = matrix4(calibration["vehicle_from_camera"])
    waymo_camera_from_optical = camera_axis_transform()
    first_world_from_optical = (
        world_from_vehicle[first_timestamp]
        @ vehicle_from_camera
        @ waymo_camera_from_optical
    )
    origin_from_world = np.linalg.inv(first_world_from_optical)

    manifest_lines: List[str] = []
    trajectory_lines: List[str] = []
    frame_map_lines: List[str] = []
    for frame_index, row in enumerate(image_rows):
        timestamp_micros = row["timestamp_micros"]
        if timestamp_micros not in world_from_vehicle:
            raise RuntimeError(f"missing vehicle pose at {timestamp_micros}")
        relative_timestamp = (timestamp_micros - first_timestamp) * 1.0e-6
        image_name = f"{frame_index:06d}.jpg"
        image_path = image_root / image_name
        image_path.write_bytes(row["image"])
        if image_path.stat().st_size == 0:
            raise RuntimeError(f"empty JPEG written: {image_path}")

        origin_from_optical = origin_from_world @ (
            world_from_vehicle[timestamp_micros]
            @ vehicle_from_camera
            @ waymo_camera_from_optical
        )
        translation = origin_from_optical[:3, 3]
        quaternion = rotation_to_quaternion(origin_from_optical[:3, :3])
        manifest_lines.append(f"{relative_timestamp:.6f} rgb/{image_name}")
        trajectory_lines.append(
            f"{relative_timestamp:.6f} "
            f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
            f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
            f"{quaternion[2]:.9f} {quaternion[3]:.9f}"
        )
        frame_map_lines.append(
            f"{frame_index} {timestamp_micros} {relative_timestamp:.6f} rgb/{image_name}"
        )

    write_lines(
        output_root / "rgb.txt",
        ["timestamp_seconds relative_image_path", "Waymo FRONT camera"],
        manifest_lines,
    )
    write_lines(
        output_root / "groundtruth_camera_tum.txt",
        ["timestamp tx ty tz qx qy qz qw", "poses relative to the first optical camera"],
        trajectory_lines,
    )
    write_lines(
        output_root / "frame_map.txt",
        ["frame_index original_timestamp_micros relative_timestamp image"],
        frame_map_lines,
    )

    metadata = {
        "segment": segment_root.name,
        "source": str(segment_root),
        "camera": calibration,
        "num_frames": len(image_rows),
        "first_timestamp_micros": first_timestamp,
        "last_timestamp_micros": timestamps[-1],
        "duration_seconds": (timestamps[-1] - first_timestamp) * 1.0e-6,
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")

    print(f"Prepared {len(image_rows)} FRONT-camera frames in {output_root}")
    print(f"Duration: {metadata['duration_seconds']:.3f} s")


if __name__ == "__main__":
    main()
