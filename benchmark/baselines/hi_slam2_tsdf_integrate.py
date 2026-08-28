#!/usr/bin/env python3
"""HI-SLAM2 rendered-depth TSDF integration.

Adapted from ``tsdf_integrate.py`` at HI-SLAM2 commit
76c833c7d8ed474f0f3ba18056c1803e032a537f. The CPU fallback and explicit
trajectory argument make the same integration usable in isolated benchmark
containers without changing its fusion rule.
"""

from __future__ import annotations

import argparse
import os
import time
from glob import glob

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import trange


def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def load_intrinsic_extrinsic(result, stamps, traj_file):
    c = np.load(f"{result}/intrinsics.npy")
    intrinsic = o3d.core.Tensor(
        [[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]],
        dtype=o3d.core.Dtype.Float64,
    )
    poses = np.atleast_2d(np.loadtxt(traj_file))
    poses = [
        np.linalg.inv(to_se3_matrix(poses[int(stamp)])) for stamp in stamps
    ]
    poses = [
        o3d.core.Tensor(pose, dtype=o3d.core.Dtype.Float64)
        for pose in poses
    ]
    return intrinsic, poses


def integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args):
    n_files = len(depth_file_names)
    device = o3d.core.Device(
        "cuda:0" if o3d.core.cuda.is_available() else "cpu:0"
    )
    print(f"Using Open3D device: {device}")

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(
            o3d.core.float32,
            o3d.core.float32,
            o3d.core.float32,
        ),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=50000,
        device=device,
    )

    start = time.time()
    for i in trange(n_files, desc="Integration progress"):
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)
        pose = extrinsic[i]
        dep = (
            cv2.imread(depth_file_names[i], cv2.IMREAD_ANYDEPTH)
            / args.depth_scale
        )
        if dep.min() >= args.depth_max:
            continue
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth,
            intrinsic,
            pose,
            args.depth_scale,
            args.depth_max,
        )
        vbg.integrate(
            frustum_block_coords,
            depth,
            color,
            intrinsic,
            pose,
            args.depth_scale,
            args.depth_max,
        )

    print(f"Integration took {time.time() - start:.2f} seconds")
    return vbg


def main():
    parser = argparse.ArgumentParser(
        description="Integrate rendered depth maps into a TSDF"
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--voxel_size", type=float, default=0.03)
    parser.add_argument("--depth_scale", type=float, default=6553.5)
    parser.add_argument("--depth_max", type=float, default=5.0)
    parser.add_argument("--weight", type=float, default=[1], nargs="+")
    parser.add_argument("--iteration", default="after_opt")
    parser.add_argument("--traj", default=None)
    parser.add_argument("--output_tag", default=None)
    args = parser.parse_args()

    depth_file_names = sorted(
        glob(f"{args.result}/renders/depth_{args.iteration}/*")
    )
    color_file_names = sorted(
        glob(f"{args.result}/renders/image_{args.iteration}/*")
    )
    if not depth_file_names or len(depth_file_names) != len(color_file_names):
        raise RuntimeError(
            "Expected matching non-empty rendered depth and color sequences"
        )
    stamps = [float(os.path.basename(path)[:-4]) for path in color_file_names]
    print(
        f"Found {len(depth_file_names)} depth maps and "
        f"{len(color_file_names)} color images"
    )

    traj_file = args.traj or f"{args.result}/traj_full.txt"
    intrinsic, extrinsic = load_intrinsic_extrinsic(
        args.result, stamps, traj_file
    )
    vbg = integrate(
        depth_file_names,
        color_file_names,
        intrinsic,
        extrinsic,
        args,
    )

    output_tag = args.output_tag
    if output_tag is None:
        output_tag = "" if args.iteration == "after_opt" else f"{args.iteration}_"
    for weight in args.weight:
        mesh = vbg.extract_triangle_mesh(weight_threshold=weight).to_legacy()
        output = f"{args.result}/tsdf_mesh_{output_tag}w{weight:.1f}.ply"
        o3d.io.write_triangle_mesh(output, mesh)
        print(f"TSDF saved to {output}")


if __name__ == "__main__":
    main()
