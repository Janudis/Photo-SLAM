#!/usr/bin/env python3
"""Align a Replica GT mesh into a Photo-SLAM/ORB-SLAM trajectory frame.

This is the same trajectory-center Sim3 pre-alignment used by mesh_eval:
estimate slam_T_gt by first fitting slam centers -> Replica GT centers with
Umeyama, then invert that similarity and apply it to the GT mesh vertices.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np


PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_replica_binary_ply(path: Path):
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break

        if len(header) < 2 or header[0] != "ply":
            raise RuntimeError(f"Not a PLY file: {path}")
        if header[1] != "format binary_little_endian 1.0":
            raise RuntimeError(f"Only binary_little_endian PLY is supported: {path}")

        vertex_count = 0
        face_count = 0
        vertex_props: list[tuple[str, str]] = []
        face_count_type = None
        face_index_type = None
        element = None
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    vertex_count = int(parts[2])
                elif element == "face":
                    face_count = int(parts[2])
            elif element == "vertex" and len(parts) == 3 and parts[0] == "property":
                dtype = PLY_DTYPES.get(parts[1])
                if dtype is None:
                    raise RuntimeError(f"Unsupported vertex property type: {parts[1]}")
                vertex_props.append((parts[2], dtype))
            elif (
                element == "face"
                and len(parts) == 5
                and parts[0] == "property"
                and parts[1] == "list"
            ):
                face_count_type = parts[2]
                face_index_type = parts[3]

        if vertex_count <= 0 or face_count <= 0 or not vertex_props:
            raise RuntimeError("PLY is missing vertices or faces")
        if face_count_type not in ("uchar", "uint8") or face_index_type not in ("int", "int32"):
            raise RuntimeError("Only property list uchar int vertex_indices is supported")

        vertex_dtype = np.dtype(vertex_props)
        vertex_records = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
        if vertex_records.shape[0] != vertex_count:
            raise RuntimeError("Could not read all vertices")
        for name in ("x", "y", "z"):
            if name not in vertex_records.dtype.names:
                raise RuntimeError(f"PLY missing vertex property {name}")

        vertices = np.stack(
            [vertex_records["x"], vertex_records["y"], vertex_records["z"]],
            axis=1,
        ).astype(np.float64)

        colors = None
        if all(name in vertex_records.dtype.names for name in ("red", "green", "blue")):
            colors = np.stack(
                [vertex_records["red"], vertex_records["green"], vertex_records["blue"]],
                axis=1,
            ).astype(np.uint8, copy=False)

        faces = []
        for _ in range(face_count):
            count_raw = f.read(1)
            if not count_raw:
                raise RuntimeError("Unexpected EOF while reading faces")
            count = count_raw[0]
            idx_raw = f.read(4 * count)
            if len(idx_raw) != 4 * count:
                raise RuntimeError("Unexpected EOF while reading face indices")
            idx = struct.unpack("<" + "i" * count, idx_raw)
            if count < 3:
                continue
            for j in range(1, count - 1):
                faces.append((idx[0], idx[j], idx[j + 1]))

    faces_np = np.asarray(faces, dtype=np.int32)
    if faces_np.size == 0:
        raise RuntimeError("PLY has no triangulatable faces")
    return vertices, faces_np, colors


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != vertices.shape[0]:
            colors = None

    with path.open("wb") as f:
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {vertices.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if colors is not None:
            header += [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        header += [
            f"element face {faces.shape[0]}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        f.write(("\n".join(header) + "\n").encode("ascii"))

        if colors is None:
            vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            vertex_records = np.empty(vertices.shape[0], dtype=vertex_dtype)
        else:
            vertex_dtype = np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            )
            vertex_records = np.empty(vertices.shape[0], dtype=vertex_dtype)
            vertex_records["red"] = colors[:, 0]
            vertex_records["green"] = colors[:, 1]
            vertex_records["blue"] = colors[:, 2]
        vertex_records["x"] = vertices[:, 0]
        vertex_records["y"] = vertices[:, 1]
        vertex_records["z"] = vertices[:, 2]
        vertex_records.tofile(f)

        for tri in faces:
            f.write(struct.pack("<Biii", 3, int(tri[0]), int(tri[1]), int(tri[2])))


def load_replica_traj_centers(path: Path) -> np.ndarray:
    centers = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 16:
                continue
            T = np.asarray(vals[:16], dtype=np.float64).reshape(4, 4)
            centers.append(T[:3, 3].copy())
    if not centers:
        raise RuntimeError(f"No valid Replica trajectory poses: {path}")
    return np.asarray(centers, dtype=np.float64)


def load_tum_centers(path: Path) -> np.ndarray:
    centers = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 8:
                continue
            centers.append(vals[1:4])
    if not centers:
        raise RuntimeError(f"No valid TUM poses: {path}")
    return np.asarray(centers, dtype=np.float64)


def estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 4 or src.shape[0] != dst.shape[0]:
        raise RuntimeError("Need at least 4 paired source/destination centers")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src[None, :]
    yd = dst - mu_dst[None, :]
    var_src = np.mean(np.sum(xs * xs, axis=1))
    if var_src <= 1.0e-15:
        raise RuntimeError("Degenerate source trajectory")

    cov = (yd.T @ xs) / float(src.shape[0])
    U, singular, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0.0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = float(np.sum(singular * np.diag(S)) / var_src)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Invalid estimated scale")
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_mesh", required=True, type=Path)
    parser.add_argument("--gt_traj", required=True, type=Path)
    parser.add_argument("--slam_tum", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_pairs", type=int, default=2000)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    gt_centers_all = load_replica_traj_centers(args.gt_traj)
    slam_centers_all = load_tum_centers(args.slam_tum)
    n = min(gt_centers_all.shape[0], slam_centers_all.shape[0])
    if n < 4:
        raise RuntimeError("Not enough trajectory correspondences")

    stride = max(1, args.stride)
    idx = np.arange(0, n, stride, dtype=np.int64)
    if args.max_pairs > 0 and idx.shape[0] > args.max_pairs:
        idx = idx[: args.max_pairs]
    if idx.shape[0] < 4:
        raise RuntimeError("Not enough trajectory correspondences after stride/max_pairs")

    # Same pre-alignment direction as mesh_eval: slam/recon centers -> GT centers.
    slam_to_gt_scale, slam_to_gt_R, slam_to_gt_t = estimate_similarity_umeyama(
        slam_centers_all[idx],
        gt_centers_all[idx],
    )

    # We need GT mesh in SLAM frame for Rerun/live SDF debugging.
    gt_to_slam_scale = 1.0 / slam_to_gt_scale
    gt_to_slam_R = slam_to_gt_R.T
    gt_to_slam_t = -gt_to_slam_scale * (gt_to_slam_R @ slam_to_gt_t)

    vertices, faces, colors = load_replica_binary_ply(args.gt_mesh)
    aligned_vertices = gt_to_slam_scale * (vertices @ gt_to_slam_R.T) + gt_to_slam_t[None, :]
    write_binary_ply(args.out, aligned_vertices, faces, colors)

    metadata_path = args.metadata or args.out.with_suffix(args.out.suffix + ".json")
    metadata = {
        "gt_mesh": str(args.gt_mesh),
        "gt_traj": str(args.gt_traj),
        "slam_tum": str(args.slam_tum),
        "out": str(args.out),
        "pairs": int(idx.shape[0]),
        "stride": int(stride),
        "slam_to_gt_scale": float(slam_to_gt_scale),
        "slam_to_gt_R": slam_to_gt_R.tolist(),
        "slam_to_gt_t": slam_to_gt_t.tolist(),
        "gt_to_slam_scale": float(gt_to_slam_scale),
        "gt_to_slam_R": gt_to_slam_R.tolist(),
        "gt_to_slam_t": gt_to_slam_t.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[align_replica_gt_mesh_to_slam] wrote {args.out}")
    print(f"[align_replica_gt_mesh_to_slam] metadata {metadata_path}")
    print(
        "[align_replica_gt_mesh_to_slam] "
        f"pairs={idx.shape[0]} slam_to_gt_scale={slam_to_gt_scale:.8f} "
        f"gt_to_slam_scale={gt_to_slam_scale:.8f}"
    )


if __name__ == "__main__":
    main()
