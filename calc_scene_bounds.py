#!/usr/bin/env python3
import argparse, os, numpy as np

def load_tum_groundtruth(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 8:
                continue
            # timestamp, tx ty tz qx qy qz qw
            try:
                ts = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                rows.append((ts, tx, ty, tz))
            except ValueError:
                pass
    if not rows:
        raise RuntimeError(f"No valid poses found in {path}")
    rows.sort(key=lambda r: r[0])  # sort by timestamp
    poses = np.asarray([[r[1], r[2], r[3]] for r in rows], dtype=np.float32)
    return poses

def bbox_stats(min_xyz, max_xyz):
    size = np.maximum(max_xyz - min_xyz, 0.0)
    center = 0.5 * (min_xyz + max_xyz)
    inside_extent = float(np.max(size))              # SVRaster scalar extent (no inflation)
    diagonal      = float(np.linalg.norm(size))      # bbox diagonal
    return size, center, inside_extent, diagonal

def inflate_bounds(min_xyz, max_xyz, pad_scale=1.05, pad_m=0.05):
    size = np.maximum(max_xyz - min_xyz, 0.0)
    ctr  = 0.5 * (min_xyz + max_xyz)
    half = 0.5 * size
    half = half * pad_scale + pad_m
    new_min = ctr - half
    new_max = ctr + half
    return new_min, new_max

def path_length(poses):
    diffs = poses[1:] - poses[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())

def main():
    ap = argparse.ArgumentParser(description="Bounds & trajectory from TUM groundtruth.txt")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pad-scale", type=float, default=1.10)
    ap.add_argument("--pad-m", type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    poses = load_tum_groundtruth(args.gt)   # (N,3)

    # --- RAW (uninflated) bbox over camera centers ---
    raw_min = poses.min(axis=0)
    raw_max = poses.max(axis=0)
    raw_size, raw_ctr, raw_extent, raw_diag = bbox_stats(raw_min, raw_max)

    # --- INFLATED bbox (what you'll pass to SVRaster) ---
    inf_min, inf_max = inflate_bounds(raw_min, raw_max, args.pad_scale, args.pad_m)
    inf_size, inf_ctr, inf_extent, inf_diag = bbox_stats(inf_min, inf_max)

    # --- Trajectory length (what TUM shows on the site) ---
    traj_len = path_length(poses)

    # Save inflated bounding for SVRaster
    bounding = np.stack([inf_min, inf_max], axis=0).astype(np.float32)
    np.save(os.path.join(args.out_dir, "offline_bounding.npy"), bounding)
    with open(os.path.join(args.out_dir, "offline_bounding.txt"), "w") as f:
        f.write("# RAW bbox (no padding)\n")
        f.write(f"raw_min: {raw_min.tolist()}\nraw_max: {raw_max.tolist()}\n")
        f.write(f"raw_center: {raw_ctr.tolist()}\n")
        f.write(f"raw_inside_extent: {raw_extent:.6f}\n")
        f.write(f"raw_diagonal: {raw_diag:.6f}\n")
        f.write("# Inflated bbox (pad applied)\n")
        f.write(f"inf_min: {inf_min.tolist()}\ninf_max: {inf_max.tolist()}\n")
        f.write(f"inf_center: {inf_ctr.tolist()}\n")
        f.write(f"inf_inside_extent: {inf_extent:.6f}\n")
        f.write(f"inf_diagonal: {inf_diag:.6f}\n")
        f.write(f"# trajectory_length: {traj_len:.6f}\n")

    # Console summary
    print("=== RAW bbox (no padding) ===")
    print(f"min: {raw_min}  max: {raw_max}")
    print(f"center: {raw_ctr}")
    print(f"inside_extent (max side): {raw_extent:.6f}")
    print(f"diagonal:                {raw_diag:.6f}")
    print("\n=== Inflated bbox (SVRaster input) ===")
    print(f"min: {inf_min}  max: {inf_max}")
    print(f"center: {inf_ctr}")
    print(f"inside_extent (max side): {inf_extent:.6f}")
    print(f"diagonal:                {inf_diag:.6f}")
    print("\n=== Trajectory ===")
    print(f"path length: {traj_len:.6f} m")

if __name__ == "__main__":
    main()
