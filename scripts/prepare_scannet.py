#!/usr/bin/env python3
"""Export a ScanNet .sens sequence for the Photo-SLAM RGB-D runner.

The binary layout and decompression rules follow ScanNet's official
SensReader/python/SensorData.py. Frames are streamed instead of retaining the
complete scan in memory.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np


COLOR_COMPRESSION = {0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}
HI_SLAM2_CROP_BORDER = 12
HI_SLAM2_TARGET_PIXELS = 341 * 640
HI_SLAM2_PREPARE_MARKER = ".hislam2_scannet_preprocess_v1"


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"unexpected end of .sens file: wanted {size} bytes")
    return data


def unpack_one(handle, fmt):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, read_exact(handle, size))[0]


def read_matrix4(handle):
    return np.asarray(
        struct.unpack("<16f", read_exact(handle, 16 * 4)),
        dtype=np.float32,
    ).reshape(4, 4)


def write_matrix(path, matrix):
    np.savetxt(path, matrix, fmt="%.9f")


def rotation_matrix_to_quaternion_xyzw(rotation):
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
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
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
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("invalid camera rotation in .sens pose")
    return quaternion / norm


def decode_color(data, compression, width, height):
    if compression in ("jpeg", "png"):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {compression} color frame")
        return image
    if compression == "raw":
        expected = width * height * 3
        if len(data) != expected:
            raise ValueError(f"raw color frame has {len(data)} bytes; expected {expected}")
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported ScanNet color compression: {compression}")


def decode_depth(data, compression, width, height):
    if compression == "zlib_ushort":
        data = zlib.decompress(data)
    elif compression != "raw_ushort":
        raise ValueError(f"unsupported ScanNet depth compression: {compression}")
    expected = width * height
    depth = np.frombuffer(data, dtype="<u2")
    if depth.size != expected:
        raise ValueError(f"depth frame has {depth.size} pixels; expected {expected}")
    return depth.reshape(height, width)


def hi_slam2_target_size(width, height):
    scale = math.sqrt(HI_SLAM2_TARGET_PIXELS / float(width * height))
    target_height = int(height * scale)
    target_width = int(width * scale)
    target_height -= target_height % 8
    target_width -= target_width % 8
    return target_width, target_height


def write_orb_config(path, intrinsic, width, height, fps, depth_shift):
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    camera_fps = int(round(fps))
    config = f'''%YAML:1.0

File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: {fx:.9f}
Camera1.fy: {fy:.9f}
Camera1.cx: {cx:.9f}
Camera1.cy: {cy:.9f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.k3: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0

Camera.fps: {camera_fps}
Camera.RGB: 1
Camera.width: {width}
Camera.height: {height}

Stereo.ThDepth: 40.0
Stereo.b: 0.08
RGBD.DepthMapFactor: {depth_shift:.6f}

loopClosing: 1

ORBextractor.nFeatures: 1600
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''
    path.write_text(config, encoding="utf-8")


def parse_header(handle):
    version = unpack_one(handle, "I")
    if version != 4:
        raise ValueError(f"unsupported ScanNet .sens version {version}; expected 4")
    name_size = unpack_one(handle, "Q")
    sensor_name = read_exact(handle, name_size).decode("utf-8", errors="replace")
    intrinsic_color = read_matrix4(handle)
    extrinsic_color = read_matrix4(handle)
    intrinsic_depth = read_matrix4(handle)
    extrinsic_depth = read_matrix4(handle)
    color_code = unpack_one(handle, "i")
    depth_code = unpack_one(handle, "i")
    if color_code not in COLOR_COMPRESSION or depth_code not in DEPTH_COMPRESSION:
        raise ValueError(
            f"unsupported compression codes: color={color_code}, depth={depth_code}")
    header = {
        "version": version,
        "sensor_name": sensor_name,
        "intrinsic_color": intrinsic_color,
        "extrinsic_color": extrinsic_color,
        "intrinsic_depth": intrinsic_depth,
        "extrinsic_depth": extrinsic_depth,
        "color_compression": COLOR_COMPRESSION[color_code],
        "depth_compression": DEPTH_COMPRESSION[depth_code],
        "color_width": unpack_one(handle, "I"),
        "color_height": unpack_one(handle, "I"),
        "depth_width": unpack_one(handle, "I"),
        "depth_height": unpack_one(handle, "I"),
        "depth_shift": unpack_one(handle, "f"),
        "num_frames": unpack_one(handle, "Q"),
    }
    return header


def export_scan(args):
    sens_path = args.sens.resolve()
    output_dir = args.output_dir.resolve()
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "pose"
    intrinsic_dir = output_dir / "intrinsic"
    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    association_lines = ["# timestamp rgb_path timestamp depth_path"]
    trajectory_lines = ["# timestamp tx ty tz qx qy qz qw"]
    exported_frames = []

    with sens_path.open("rb") as handle:
        header = parse_header(handle)
        crop_border = HI_SLAM2_CROP_BORDER if args.hi_slam2_preprocess else 0
        cropped_color_width = header["color_width"] - 2 * crop_border
        cropped_color_height = header["color_height"] - 2 * crop_border
        if cropped_color_width <= 0 or cropped_color_height <= 0:
            raise ValueError("HI-SLAM2 crop border is larger than the color frame")

        if args.hi_slam2_preprocess:
            target_width, target_height = hi_slam2_target_size(
                cropped_color_width, cropped_color_height)
        else:
            target_width, target_height = args.width, args.height

        scale_x = target_width / float(cropped_color_width)
        scale_y = target_height / float(cropped_color_height)
        target_intrinsic = np.array(header["intrinsic_color"], copy=True)
        target_intrinsic[0, 2] -= crop_border
        target_intrinsic[1, 2] -= crop_border
        target_intrinsic[0, :] *= scale_x
        target_intrinsic[1, :] *= scale_y
        target_intrinsic[2, 2] = 1.0

        depth_crop_x = int(round(
            crop_border * header["depth_width"] / float(header["color_width"])))
        depth_crop_y = int(round(
            crop_border * header["depth_height"] / float(header["color_height"])))

        write_matrix(intrinsic_dir / "intrinsic_color.txt", header["intrinsic_color"])
        write_matrix(intrinsic_dir / "extrinsic_color.txt", header["extrinsic_color"])
        write_matrix(intrinsic_dir / "intrinsic_depth.txt", header["intrinsic_depth"])
        write_matrix(intrinsic_dir / "extrinsic_depth.txt", header["extrinsic_depth"])
        write_matrix(intrinsic_dir / "intrinsic_color_target.txt", target_intrinsic)
        write_orb_config(
            output_dir / "orb_slam3_rgbd.yaml",
            target_intrinsic,
            target_width,
            target_height,
            args.fps / args.frame_stride,
            header["depth_shift"],
        )

        print(
            f"[prepare_scannet] {sens_path.name}: frames={header['num_frames']} "
            f"color={header['color_width']}x{header['color_height']} "
            f"depth={header['depth_width']}x{header['depth_height']} "
            f"target={target_width}x{target_height} "
            f"stride={args.frame_stride} "
            f"hi_slam2_preprocess={int(args.hi_slam2_preprocess)}")

        for frame_index in range(header["num_frames"]):
            camera_to_world = read_matrix4(handle)
            timestamp_color = unpack_one(handle, "Q")
            timestamp_depth = unpack_one(handle, "Q")
            color_size = unpack_one(handle, "Q")
            depth_size = unpack_one(handle, "Q")
            color_data = read_exact(handle, color_size)
            depth_data = read_exact(handle, depth_size)
            if frame_index % args.frame_stride != 0:
                continue

            color = decode_color(
                color_data,
                header["color_compression"],
                header["color_width"],
                header["color_height"],
            )
            depth = decode_depth(
                depth_data,
                header["depth_compression"],
                header["depth_width"],
                header["depth_height"],
            )
            if crop_border:
                color = color[
                    crop_border:color.shape[0] - crop_border,
                    crop_border:color.shape[1] - crop_border]
            if depth_crop_x or depth_crop_y:
                y_end = depth.shape[0] - depth_crop_y if depth_crop_y else depth.shape[0]
                x_end = depth.shape[1] - depth_crop_x if depth_crop_x else depth.shape[1]
                depth = depth[depth_crop_y:y_end, depth_crop_x:x_end]
            if color.shape[1] != target_width or color.shape[0] != target_height:
                color = cv2.resize(
                    color, (target_width, target_height), interpolation=cv2.INTER_AREA)
            if depth.shape[1] != target_width or depth.shape[0] != target_height:
                depth = cv2.resize(
                    depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

            stem = f"{frame_index:06d}"
            color_path = color_dir / f"{stem}.jpg"
            depth_path = depth_dir / f"{stem}.png"
            pose_path = pose_dir / f"{stem}.txt"
            if args.overwrite or not color_path.exists():
                if not cv2.imwrite(str(color_path), color, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(f"failed to write {color_path}")
            if args.overwrite or not depth_path.exists():
                if not cv2.imwrite(str(depth_path), depth):
                    raise IOError(f"failed to write {depth_path}")
            write_matrix(pose_path, camera_to_world)

            timestamp = frame_index / args.fps
            association_lines.append(
                f"{timestamp:.9f} color/{color_path.name} "
                f"{timestamp:.9f} depth/{depth_path.name}")
            if np.isfinite(camera_to_world).all():
                quaternion = rotation_matrix_to_quaternion_xyzw(camera_to_world[:3, :3])
                translation = camera_to_world[:3, 3]
                trajectory_lines.append(
                    f"{timestamp:.9f} "
                    f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                    f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
                    f"{quaternion[2]:.9f} {quaternion[3]:.9f}")
            exported_frames.append({
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "sensor_timestamp_color": timestamp_color,
                "sensor_timestamp_depth": timestamp_depth,
            })
            if len(exported_frames) % 250 == 0:
                print(f"[prepare_scannet] exported {len(exported_frames)} frames")

    (output_dir / "association.txt").write_text(
        "\n".join(association_lines) + "\n", encoding="utf-8")
    (output_dir / "groundtruth_tum.txt").write_text(
        "\n".join(trajectory_lines) + "\n", encoding="utf-8")
    metadata = {
        "sens_file": str(sens_path),
        "sensor_name": header["sensor_name"],
        "source_frames": header["num_frames"],
        "exported_frame_count": len(exported_frames),
        "frame_stride": args.frame_stride,
        "fps": args.fps,
        "effective_fps": args.fps / args.frame_stride,
        "width": target_width,
        "height": target_height,
        "hi_slam2_preprocess": args.hi_slam2_preprocess,
        "crop_border": crop_border,
        "depth_shift": header["depth_shift"],
        "color_compression": header["color_compression"],
        "depth_compression": header["depth_compression"],
        "target_intrinsic": target_intrinsic.tolist(),
        "frames": exported_frames,
    }
    (output_dir / "scannet_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.hi_slam2_preprocess:
        (output_dir / HI_SLAM2_PREPARE_MARKER).write_text(
            "HI-SLAM2 ScanNet crop/resize preprocessing v1\n", encoding="utf-8")
    print(
        f"[prepare_scannet] ready: {output_dir} "
        f"({len(exported_frames)} RGB-D frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Stream a ScanNet .sens sequence into Photo-SLAM RGB-D inputs")
    parser.add_argument("--sens", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--hi-slam2-preprocess",
        action="store_true",
        help="apply HI-SLAM2's ScanNet 12-pixel crop and target-area resize")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.sens.is_file():
        parser.error(f".sens file does not exist: {args.sens}")
    if args.output_dir is None:
        args.output_dir = args.sens.parent
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.width < 1 or args.height < 1 or args.fps <= 0.0:
        parser.error("width, height, and fps must be positive")
    export_scan(args)


if __name__ == "__main__":
    main()
