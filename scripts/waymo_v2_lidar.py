#!/usr/bin/env python3

"""Project Waymo v2 LiDAR range images into a FRONT-camera depth map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq


LIDAR_RANGE_RETURN1 = "[LiDARComponent].range_image_return1"
LIDAR_RANGE_RETURN2 = "[LiDARComponent].range_image_return2"
LIDAR_EXTRINSIC = "[LiDARCalibrationComponent].extrinsic.transform"
LIDAR_INCLINATION_MIN = "[LiDARCalibrationComponent].beam_inclination.min"
LIDAR_INCLINATION_MAX = "[LiDARCalibrationComponent].beam_inclination.max"
LIDAR_INCLINATIONS = "[LiDARCalibrationComponent].beam_inclination.values"
LIDAR_POSE_RETURN1 = "[LiDARPoseComponent].range_image_return1"
TOP_LIDAR_ID = 1


@dataclass(frozen=True)
class LidarCalibration:
    extrinsic: np.ndarray
    inclination_min: float
    inclination_max: float
    inclinations: np.ndarray


def _find_component(segment_root: Path, component: str) -> Path:
    matches = sorted((segment_root / component).glob("*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {component} parquet under {segment_root}, found {len(matches)}"
        )
    return matches[0]


def _matrix4(values: Iterable[float]) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"expected a 4x4 transform, received {matrix.size} values")
    return matrix.reshape(4, 4)


def _iter_rows(
    path: Path,
    columns: Sequence[str],
    batch_size: int,
) -> Iterator[Dict[str, object]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            yield {name: values[name][index] for name in columns}


def _list_array_views(array: object) -> tuple[np.ndarray, np.ndarray]:
    offsets = array.offsets.to_numpy(zero_copy_only=True)
    values = array.values.to_numpy(zero_copy_only=True)
    return offsets, values


def _iter_lidar_range_rows(
    path: Path,
) -> Iterator[tuple[int, int, tuple[np.ndarray, tuple[int, ...]], tuple[np.ndarray, tuple[int, ...]]]]:
    columns = [
        "key.frame_timestamp_micros",
        "key.laser_name",
        f"{LIDAR_RANGE_RETURN1}.values",
        f"{LIDAR_RANGE_RETURN1}.shape",
        f"{LIDAR_RANGE_RETURN2}.values",
        f"{LIDAR_RANGE_RETURN2}.shape",
    ]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=5, columns=columns):
        timestamps = batch.column(0).to_numpy(zero_copy_only=True)
        laser_ids = batch.column(1).to_numpy(zero_copy_only=True)
        return1_offsets, return1_values = _list_array_views(batch.column(2))
        return2_offsets, return2_values = _list_array_views(batch.column(4))
        return1_shapes = batch.column(3).values.to_numpy(zero_copy_only=True).reshape(-1, 3)
        return2_shapes = batch.column(5).values.to_numpy(zero_copy_only=True).reshape(-1, 3)
        for index in range(batch.num_rows):
            return1 = return1_values[return1_offsets[index] : return1_offsets[index + 1]]
            return2 = return2_values[return2_offsets[index] : return2_offsets[index + 1]]
            yield (
                int(timestamps[index]),
                int(laser_ids[index]),
                (return1, tuple(int(value) for value in return1_shapes[index])),
                (return2, tuple(int(value) for value in return2_shapes[index])),
            )


def _iter_top_lidar_pose_rows(
    path: Path,
) -> Iterator[tuple[int, int, np.ndarray]]:
    columns = [
        "key.frame_timestamp_micros",
        "key.laser_name",
        f"{LIDAR_POSE_RETURN1}.values",
        f"{LIDAR_POSE_RETURN1}.shape",
    ]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4, columns=columns):
        timestamps = batch.column(0).to_numpy(zero_copy_only=True)
        laser_ids = batch.column(1).to_numpy(zero_copy_only=True)
        offsets, values = _list_array_views(batch.column(2))
        shapes = batch.column(3).values.to_numpy(zero_copy_only=True).reshape(-1, 3)
        for index in range(batch.num_rows):
            shape = tuple(int(value) for value in shapes[index])
            pose = values[offsets[index] : offsets[index + 1]].reshape(shape)
            yield int(timestamps[index]), int(laser_ids[index]), pose


def _load_lidar_calibrations(segment_root: Path) -> Dict[int, LidarCalibration]:
    path = _find_component(segment_root, "lidar_calibration")
    columns = [
        "key.laser_name",
        LIDAR_EXTRINSIC,
        LIDAR_INCLINATION_MIN,
        LIDAR_INCLINATION_MAX,
        LIDAR_INCLINATIONS,
    ]
    calibrations: Dict[int, LidarCalibration] = {}
    for row in _iter_rows(path, columns, batch_size=8):
        laser_id = int(row["key.laser_name"])
        inclinations = row[LIDAR_INCLINATIONS] or []
        calibrations[laser_id] = LidarCalibration(
            extrinsic=_matrix4(row[LIDAR_EXTRINSIC]),
            inclination_min=float(row[LIDAR_INCLINATION_MIN]),
            inclination_max=float(row[LIDAR_INCLINATION_MAX]),
            inclinations=np.asarray(inclinations, dtype=np.float64),
        )
    return calibrations


class _TopLidarPoseStream:
    def __init__(self, segment_root: Path) -> None:
        path = _find_component(segment_root, "lidar_pose")
        self._rows = _iter_top_lidar_pose_rows(path)
        self._next = next(self._rows, None)

    def get(self, timestamp_micros: int) -> np.ndarray | None:
        while self._next is not None:
            row_timestamp = self._next[0]
            if row_timestamp >= timestamp_micros:
                break
            self._next = next(self._rows, None)
        if self._next is None:
            return None
        if self._next[0] != timestamp_micros:
            return None
        _, laser_id, pose = self._next
        self._next = next(self._rows, None)
        if laser_id != TOP_LIDAR_ID:
            return None
        return pose


def _beam_inclinations(calibration: LidarCalibration, height: int) -> np.ndarray:
    if calibration.inclinations.size:
        if calibration.inclinations.size != height:
            raise ValueError(
                f"LiDAR calibration has {calibration.inclinations.size} beam angles "
                f"for a {height}-row range image"
            )
        inclinations = calibration.inclinations
    else:
        ratio = (0.5 + np.arange(height, dtype=np.float64)) / float(height)
        inclinations = calibration.inclination_min + ratio * (
            calibration.inclination_max - calibration.inclination_min
        )
    return inclinations[::-1].copy()


def _range_image_to_vehicle(
    range_image: np.ndarray,
    calibration: LidarCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = range_image.shape
    ranges = range_image[..., 0].astype(np.float64, copy=False)
    valid = np.isfinite(ranges) & (ranges > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64), valid

    inclination = _beam_inclinations(calibration, height)[rows]
    ratios = (width - cols.astype(np.float64) - 0.5) / float(width)
    azimuth_correction = np.arctan2(
        calibration.extrinsic[1, 0], calibration.extrinsic[0, 0]
    )
    azimuth = (ratios * 2.0 - 1.0) * np.pi - azimuth_correction
    distance = ranges[rows, cols]
    cos_inclination = np.cos(inclination)
    points_lidar = np.stack(
        (
            np.cos(azimuth) * cos_inclination * distance,
            np.sin(azimuth) * cos_inclination * distance,
            np.sin(inclination) * distance,
        ),
        axis=1,
    )
    rotation = calibration.extrinsic[:3, :3]
    translation = calibration.extrinsic[:3, 3]
    points_vehicle = points_lidar @ rotation.T + translation
    return points_vehicle, valid


def _apply_top_lidar_motion_compensation(
    points_vehicle: np.ndarray,
    valid_range_mask: np.ndarray,
    pixel_pose: np.ndarray,
    world_from_vehicle: np.ndarray,
) -> np.ndarray:
    if pixel_pose.shape[:2] != valid_range_mask.shape or pixel_pose.shape[-1] != 6:
        raise ValueError("TOP LiDAR pose image shape does not match its range image")
    pose = pixel_pose[valid_range_mask]
    roll, pitch, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    x, y, z = points_vehicle[:, 0], points_vehicle[:, 1], points_vehicle[:, 2]

    # Waymo constructs Rz(yaw) * Ry(pitch) * Rx(roll).
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    y_rx = cos_roll * y - sin_roll * z
    z_rx = sin_roll * y + cos_roll * z
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    x_ry = cos_pitch * x + sin_pitch * z_rx
    z_ry = -sin_pitch * x + cos_pitch * z_rx
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x_world = cos_yaw * x_ry - sin_yaw * y_rx
    y_world = sin_yaw * x_ry + cos_yaw * y_rx
    points_world = np.stack((x_world, y_world, z_ry), axis=1) + pose[:, 3:6]

    vehicle_from_world = np.linalg.inv(world_from_vehicle)
    return (
        points_world @ vehicle_from_world[:3, :3].T
        + vehicle_from_world[:3, 3]
    )


def _project_to_front_depth(
    points_vehicle: np.ndarray,
    depth: np.ndarray,
    camera: Mapping[str, object],
) -> None:
    if points_vehicle.size == 0:
        return
    vehicle_from_camera = _matrix4(camera["vehicle_from_camera"])
    waymo_camera_from_optical = np.eye(4, dtype=np.float64)
    waymo_camera_from_optical[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    optical_from_vehicle = np.linalg.inv(
        vehicle_from_camera @ waymo_camera_from_optical
    )
    points_optical = (
        points_vehicle @ optical_from_vehicle[:3, :3].T
        + optical_from_vehicle[:3, 3]
    )

    z = points_optical[:, 2]
    valid = np.isfinite(points_optical).all(axis=1) & (z > 0.0)
    if not np.any(valid):
        return
    points_optical = points_optical[valid]
    z = points_optical[:, 2]
    x = points_optical[:, 0] / z
    y = points_optical[:, 1] / z
    ideal_u = float(camera["fx"]) * x + float(camera["cx"])
    ideal_v = float(camera["fy"]) * y + float(camera["cy"])
    in_ideal_fov = (
        (ideal_u >= 0.0)
        & (ideal_u < float(camera["width"]))
        & (ideal_v >= 0.0)
        & (ideal_v < float(camera["height"]))
    )
    if not np.any(in_ideal_fov):
        return
    x, y, z = x[in_ideal_fov], y[in_ideal_fov], z[in_ideal_fov]
    r2 = x * x + y * y
    radial = (
        1.0
        + float(camera["k1"]) * r2
        + float(camera["k2"]) * r2 * r2
        + float(camera["k3"]) * r2 * r2 * r2
    )
    x_distorted = (
        x * radial
        + 2.0 * float(camera["p1"]) * x * y
        + float(camera["p2"]) * (r2 + 2.0 * x * x)
    )
    y_distorted = (
        y * radial
        + float(camera["p1"]) * (r2 + 2.0 * y * y)
        + 2.0 * float(camera["p2"]) * x * y
    )
    scale_x = depth.shape[1] / float(camera["width"])
    scale_y = depth.shape[0] / float(camera["height"])
    u = np.rint(
        (float(camera["fx"]) * x_distorted + float(camera["cx"])) * scale_x
    ).astype(np.int64)
    v = np.rint(
        (float(camera["fy"]) * y_distorted + float(camera["cy"])) * scale_y
    ).astype(np.int64)
    in_image = (
        (u >= 0)
        & (u < depth.shape[1])
        & (v >= 0)
        & (v < depth.shape[0])
    )
    u, v, z = u[in_image], v[in_image], z[in_image]
    if z.size == 0:
        return
    np.minimum.at(
        depth.reshape(-1),
        v * depth.shape[1] + u,
        z.astype(np.float32),
    )


def export_front_lidar_depth(
    segment_root: Path,
    output_root: Path,
    timestamps_micros: Sequence[int],
    camera: Mapping[str, object],
    world_from_vehicle: Mapping[int, np.ndarray],
    output_width: int,
    output_height: int,
    min_depth_m: float = 1.0,
    max_depth_m: float = 75.0,
    projection_radius_px: int = 2,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Write tracking and exact mapper depth maps in one LiDAR pass."""
    if projection_radius_px < 0:
        raise ValueError("projection_radius_px must be non-negative")
    wanted_timestamps = set(int(value) for value in timestamps_micros)
    depth_root = output_root / "depth_lidar"
    mapper_depth_root = output_root / "depth_lidar_raw"
    depth_root.mkdir(parents=True, exist_ok=True)
    mapper_depth_root.mkdir(parents=True, exist_ok=True)
    calibrations = _load_lidar_calibrations(segment_root)
    pose_stream = _TopLidarPoseStream(segment_root)

    lidar_rows = _iter_lidar_range_rows(
        _find_component(segment_root, "lidar")
    )
    output_paths: Dict[int, str] = {}
    mapper_output_paths: Dict[int, str] = {}
    current_timestamp: int | None = None
    current_depth: np.ndarray | None = None
    current_top_pose: np.ndarray | None = None
    frame_index = {timestamp: index for index, timestamp in enumerate(timestamps_micros)}
    last_wanted_timestamp = max(wanted_timestamps)
    footprint_kernel = np.ones(
        (2 * projection_radius_px + 1, 2 * projection_radius_px + 1),
        dtype=np.uint8,
    )

    def finish_frame() -> None:
        if current_timestamp is None or current_depth is None:
            return
        exact_depth = current_depth.copy()
        exact_valid = np.isfinite(exact_depth)
        exact_valid &= exact_depth >= min_depth_m
        exact_valid &= exact_depth <= max_depth_m
        exact_depth[~exact_valid] = 0.0

        depth = current_depth.copy()
        if projection_radius_px > 0:
            depth = cv2.erode(depth, footprint_kernel)
        valid = np.isfinite(depth)
        valid &= depth >= min_depth_m
        valid &= depth <= max_depth_m
        depth[~valid] = 0.0
        image_name = f"{frame_index[current_timestamp]:06d}.tiff"
        mapper_output_path = mapper_depth_root / image_name
        if not cv2.imwrite(str(mapper_output_path), exact_depth):
            raise RuntimeError(
                f"failed to write exact LiDAR depth map: {mapper_output_path}"
            )
        output_path = depth_root / image_name
        if not cv2.imwrite(str(output_path), depth):
            raise RuntimeError(f"failed to write LiDAR depth map: {output_path}")
        output_paths[current_timestamp] = str(Path("depth_lidar") / image_name)
        mapper_output_paths[current_timestamp] = str(
            Path("depth_lidar_raw") / image_name
        )
        completed = len(output_paths)
        if completed == 1 or completed % 20 == 0 or completed == len(wanted_timestamps):
            print(
                f"Projected LiDAR depth: {completed}/{len(wanted_timestamps)} frames",
                flush=True,
            )

    previous_timestamp = -1
    for timestamp, laser_id, return1, return2 in lidar_rows:
        if timestamp > last_wanted_timestamp:
            break
        if timestamp < previous_timestamp:
            raise RuntimeError("Waymo LiDAR parquet rows are not timestamp ordered")
        previous_timestamp = timestamp
        if timestamp not in wanted_timestamps:
            continue
        if current_timestamp != timestamp:
            finish_frame()
            current_timestamp = timestamp
            current_depth = np.full(
                (output_height, output_width), np.inf, dtype=np.float32
            )
            current_top_pose = pose_stream.get(timestamp)

        if laser_id not in calibrations:
            raise RuntimeError(f"missing calibration for LiDAR {laser_id}")
        for values, shape in (return1, return2):
            if values.size == 0:
                continue
            range_image = values.reshape(shape)
            points_vehicle, valid_range_mask = _range_image_to_vehicle(
                range_image, calibrations[laser_id]
            )
            if (
                laser_id == TOP_LIDAR_ID
                and current_top_pose is not None
                and timestamp in world_from_vehicle
            ):
                points_vehicle = _apply_top_lidar_motion_compensation(
                    points_vehicle,
                    valid_range_mask,
                    current_top_pose,
                    world_from_vehicle[timestamp],
                )
            _project_to_front_depth(points_vehicle, current_depth, camera)

    finish_frame()
    missing = wanted_timestamps.difference(output_paths)
    if missing:
        raise RuntimeError(f"missing LiDAR depth for {len(missing)} camera frames")
    return output_paths, mapper_output_paths
