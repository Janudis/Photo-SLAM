#!/usr/bin/env python3
# scripts_planner/run_rrt.py

import os
from os.path import abspath, dirname, join
import sys
import time
import json
import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial import KDTree

# ----------------- OMPL import (nanobind build in Photo-SLAM) -----------------
try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    photo_slam_root = dirname(dirname(abspath(__file__)))  # ~/Photo-SLAM/scripts_planner -> ~/Photo-SLAM
    sys.path.insert(0, join(photo_slam_root, "third_party/ompl/build/nanobinds"))
    from ompl import base as ob
    from ompl import geometric as og

from voxelnav.voxel_collision import VoxelEVC, VoxelEMV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Voxel map container -----------------
@dataclass
class VoxelMap:
    centers: torch.Tensor      # [N,3] float32 (device)
    half_sizes: torch.Tensor   # [N] float32 (device)
    level: torch.Tensor        # [N] uint8 (device)

    centers_cpu: np.ndarray    # [N,3] float32
    half_sizes_cpu: np.ndarray # [N] float32

    kdtree: KDTree
    max_half_size: float

    def aabb(self):
        c = self.centers_cpu
        h = self.half_sizes_cpu.reshape(-1, 1)
        mn = np.min(c - h, axis=0)
        mx = np.max(c + h, axis=0)
        return mn, mx


def load_svraster_voxels_npz(
    npz_path: str,
    device: torch.device,
    *,
    max_voxels: int | None = None,
    occupancy_mode: str = "rho_center",  # "rho_center", "alpha_proxy", "ALL"
    tau_rho: float = 2.0,
    tau_alpha: float = 0.95,
) -> VoxelMap:
    data = np.load(npz_path, allow_pickle=True)

    if "vox_center" not in data or "vox_half_size" not in data:
        raise ValueError(
            f"{npz_path} must contain keys: vox_center, vox_half_size. Found keys: {list(data.keys())}"
        )

    centers_cpu = np.asarray(data["vox_center"], dtype=np.float32)  # [N,3]
    half_cpu = np.asarray(data["vox_half_size"], dtype=np.float32)
    lvl_cpu = np.asarray(data["vox_level_u8"], dtype=np.uint8) if "vox_level_u8" in data else None

    if centers_cpu.ndim != 2 or centers_cpu.shape[1] != 3:
        raise ValueError(f"vox_center must be [N,3], got {centers_cpu.shape}")

    # Accept [N], [N,1], [N,3] for half sizes
    if half_cpu.ndim == 2 and half_cpu.shape[1] == 1:
        half_cpu = half_cpu.reshape(-1)
    elif half_cpu.ndim == 2 and half_cpu.shape[1] == 3:
        half_cpu = np.max(half_cpu, axis=1)
    elif half_cpu.ndim == 1:
        pass
    else:
        raise ValueError(f"vox_half_size must be [N], [N,1], or [N,3], got {half_cpu.shape}")

    if half_cpu.shape[0] != centers_cpu.shape[0]:
        raise ValueError(f"N mismatch: vox_center {centers_cpu.shape[0]} vs vox_half_size {half_cpu.shape[0]}")

    half_cpu = np.maximum(half_cpu, 1e-6).astype(np.float32)

    # Remove extreme outliers (keeps bounds sane)
    center_med = np.median(centers_cpu, axis=0)
    dist = np.linalg.norm(centers_cpu - center_med[None, :], axis=1)
    thr = np.percentile(dist, 99.99)
    inlier = dist <= thr
    centers_cpu = centers_cpu[inlier]
    half_cpu = half_cpu[inlier]
    if lvl_cpu is not None:
        lvl_cpu = lvl_cpu[inlier]

    # ---------------- occupancy filtering ----------------
    occ_mode = occupancy_mode.lower()
    if occ_mode == "all":
        occ = np.ones((centers_cpu.shape[0],), dtype=bool)
        occ_reason = "ALL"
    elif occ_mode == "rho_center":
        if "vox_rho_center" not in data:
            raise ValueError("Need vox_rho_center in npz for occupancy_mode='rho_center'.")
        rho = np.asarray(data["vox_rho_center"], dtype=np.float32)[inlier]
        occ = rho > float(tau_rho)
        occ_reason = f"rho_center>{tau_rho:g}"
    elif occ_mode == "alpha_proxy":
        if "vox_alpha_proxy" not in data:
            raise ValueError("Need vox_alpha_proxy in npz for occupancy_mode='alpha_proxy'.")
        alp = np.asarray(data["vox_alpha_proxy"], dtype=np.float32)[inlier]
        occ = alp > float(tau_alpha)
        occ_reason = f"alpha_proxy>{tau_alpha:g}"
    else:
        raise ValueError(f"Unknown occupancy_mode={occupancy_mode}")

    centers_cpu = centers_cpu[occ]
    half_cpu = half_cpu[occ]
    if lvl_cpu is not None:
        lvl_cpu = lvl_cpu[occ]

    # Optional subsample AFTER filtering
    N = centers_cpu.shape[0]
    if max_voxels is not None and N > max_voxels:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_voxels, replace=False)
        centers_cpu = centers_cpu[idx]
        half_cpu = half_cpu[idx]
        if lvl_cpu is not None:
            lvl_cpu = lvl_cpu[idx]
        N = centers_cpu.shape[0]

    centers = torch.from_numpy(centers_cpu).to(device=device, dtype=torch.float32)
    half = torch.from_numpy(half_cpu).to(device=device, dtype=torch.float32)

    if lvl_cpu is None:
        lvl = torch.zeros((N,), dtype=torch.uint8, device=device)
    else:
        lvl = torch.from_numpy(lvl_cpu).to(device=device)

    kdtree = KDTree(centers_cpu) if N > 0 else KDTree(np.zeros((1, 3), dtype=np.float32))
    max_half = float(np.max(half_cpu)) if half_cpu.size > 0 else 0.0

    print(f"[load] {npz_path}")
    print(f"[load] inlier_thr={thr:.3f}  occupancy={occ_reason}  N_occ={N}  max_half={max_half:.6f}")

    return VoxelMap(
        centers=centers,
        half_sizes=half,
        level=lvl,
        centers_cpu=centers_cpu,
        half_sizes_cpu=half_cpu,
        kdtree=kdtree,
        max_half_size=max_half,
    )


def set_bounds_from_npz_or_voxels(space, vox: VoxelMap, npz_data, pad_ratio=0.10, pad_abs=0.05):
    if "scene_center" in npz_data and "scene_extent" in npz_data:
        c = np.asarray(npz_data["scene_center"], dtype=np.float32).reshape(3)
        e = float(np.asarray(npz_data["scene_extent"], dtype=np.float32).reshape(-1)[0])
        mn = c - 0.5 * e
        mx = c + 0.5 * e
    else:
        mn, mx = vox.aabb()

    span = mx - mn
    pad = np.maximum(pad_abs, pad_ratio * span)
    low = (mn - pad).astype(np.float32)
    high = (mx + pad).astype(np.float32)

    bounds = ob.RealVectorBounds(3)
    for i in range(3):
        bounds.setLow(i, float(low[i]))
        bounds.setHigh(i, float(high[i]))
    space.setBounds(bounds)

    print("[bounds] low =", low, "high =", high, "span =", (high - low))
    return low, high


def sample_ring_pairs(center: np.ndarray, *, n: int, radius_xy: float, radius_z: float, z_freq: float = 10.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False).astype(np.float32)
    tz = (z_freq * t).astype(np.float32)

    x0 = np.stack(
        [radius_xy * np.cos(t), radius_xy * np.sin(t), radius_z * np.sin(tz)],
        axis=-1,
    ).astype(np.float32) + center

    xf = np.stack(
        [radius_xy * np.cos(t + np.pi), radius_xy * np.sin(t + np.pi), radius_z * np.sin(tz + np.pi)],
        axis=-1,
    ).astype(np.float32) + center

    return x0, xf


def clamp_points_to_bounds(P: np.ndarray, low: np.ndarray, high: np.ndarray):
    return np.minimum(np.maximum(P, low.reshape(1, 3)), high.reshape(1, 3))


def clamp_stats(P_before: np.ndarray, P_after: np.ndarray, name: str):
    diff = np.linalg.norm(P_before - P_after, axis=1)
    n = diff.shape[0]
    n_clamped = int(np.sum(diff > 1e-6))
    mx = float(diff.max()) if n > 0 else 0.0
    mean = float(diff.mean()) if n > 0 else 0.0
    print(f"[clamp] {name}: {n_clamped}/{n} clamped, max={mx:.6g}, mean={mean:.6g}")


def path_to_list(pgeom):
    out = []
    nst = pgeom.getStateCount()
    for i in range(nst):
        s = pgeom.getState(i)
        out.append((float(s[0]), float(s[1]), float(s[2])))
    return out


def make_state(space: ob.RealVectorStateSpace, xyz: np.ndarray):
    ScopedState = getattr(ob, "ScopedState", None)
    if ScopedState is not None:
        s = ScopedState(space)
        st = s()
        st[0] = float(xyz[0])
        st[1] = float(xyz[1])
        st[2] = float(xyz[2])
        return s
    s = space.allocState()
    s[0] = float(xyz[0])
    s[1] = float(xyz[1])
    s[2] = float(xyz[2])
    return s


def make_space_information(space: ob.RealVectorStateSpace, evc: VoxelEVC, emv: VoxelEMV):
    """Fresh SI per trial to avoid lifetime issues in nanobind OMPL."""
    si = ob.SpaceInformation(space)

    class MV(ob.MotionValidator):
        def __init__(self, si_):
            super().__init__(si_)

        def checkMotion(self, s1, s2) -> bool:
            p0 = np.array([s1[0], s1[1], s1[2]], dtype=np.float32)
            p1 = np.array([s2[0], s2[1], s2[2]], dtype=np.float32)
            return bool(emv.is_valid_segment(p0, p1))

    class SVC(ob.StateValidityChecker):
        def __init__(self, si_):
            super().__init__(si_)

        def isValid(self, state) -> bool:
            p = np.array([state[0], state[1], state[2]], dtype=np.float32)
            return bool(evc.is_valid_point(p))

    mv = MV(si)
    svc = SVC(si)
    si.setMotionValidator(mv)
    si.setStateValidityChecker(svc)
    si.setup()
    return si, mv, svc


def path_length_xyz(path_list):
    if len(path_list) < 2:
        return 0.0
    P = np.asarray(path_list, dtype=np.float32)
    return float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))


def validate_path_with_checkers(path_list, evc: VoxelEVC, emv: VoxelEMV):
    """Post-check returned path using our own collision semantics."""
    if len(path_list) == 0:
        return True, True, -1

    for i, p in enumerate(path_list):
        if not bool(evc.is_valid_point(np.asarray(p, dtype=np.float32))):
            return False, True, i

    for i in range(len(path_list) - 1):
        p0 = np.asarray(path_list[i], dtype=np.float32)
        p1 = np.asarray(path_list[i + 1], dtype=np.float32)
        if not bool(emv.is_valid_segment(p0, p1)):
            return True, False, i

    return True, True, -1


def main():
    # ----------------- user params -----------------
    scene_name = "statues"
    method = "ompl"

    vox_npz = "/home/dimitris/svraster/outputs/statues_svraster/voxels_iter020000.npz"

    max_voxels = 250_000  # speed knob
    robot_radius = 0.03

    n_pairs = 100
    ring_z_freq = 10.0

    planner_range = 0.05
    planner_allowed_termination = 0.02
    short_termination_time = 5.0
    full_termination_time = short_termination_time

    occupancy_mode = "rho_center"  # "rho_center" or "alpha_proxy" or "ALL"
    tau_rho = 2.0
    tau_alpha = 0.95

    # ----------------- load NPZ + voxels -----------------
    npz_data = np.load(vox_npz, allow_pickle=True)

    if "scene_center" in npz_data:
        scene_center = np.asarray(npz_data["scene_center"], dtype=np.float32).reshape(3)
    else:
        scene_center = None

    if "inside_extent" in npz_data:
        svr_inside_extent = float(np.asarray(npz_data["inside_extent"], dtype=np.float32).reshape(-1)[0])
    else:
        svr_inside_extent = None

    if scene_center is not None:
        print("[svr] scene_center =", scene_center)
    if svr_inside_extent is not None:
        print("[svr] inside_extent =", svr_inside_extent)

    vox = load_svraster_voxels_npz(
        vox_npz,
        device=device,
        max_voxels=max_voxels,
        occupancy_mode=occupancy_mode,
        tau_rho=tau_rho,
        tau_alpha=tau_alpha,
    )

    mn, mx = vox.aabb()
    print("[aabb] mn=", mn, "mx=", mx, "span=", (mx - mn))

    if scene_center is None:
        scene_center = np.median(vox.centers_cpu, axis=0).astype(np.float32)
        print("[scene_center] median(vox) =", scene_center)
    else:
        print("[scene_center] npz.scene_center =", scene_center)

    # ----------------- OMPL space + bounds -----------------
    space = ob.RealVectorStateSpace(3)
    low_bound, high_bound = set_bounds_from_npz_or_voxels(space, vox, npz_data)

    # collision checkers
    evc = VoxelEVC(vox, robot_radius=robot_radius, device=device)
    emv = VoxelEMV(vox, robot_radius=robot_radius, device=device)

    # ----------------- ring sizing based on bounds (NO inside_extent scaling) -----------------
    span = (high_bound - low_bound).astype(np.float32)
    radius_xy = 0.35 * float(min(span[0], span[1]))
    radius_z = 0.05 * float(span[2])

    print("[ring] span =", span)
    print("[ring] radius_xy =", radius_xy, "radius_z =", radius_z, "z_freq =", ring_z_freq)

    x0_raw, xf_raw = sample_ring_pairs(
        scene_center,
        n=n_pairs,
        radius_xy=radius_xy,
        radius_z=radius_z,
        z_freq=ring_z_freq,
    )
    x0 = clamp_points_to_bounds(x0_raw, low_bound, high_bound)
    xf = clamp_points_to_bounds(xf_raw, low_bound, high_bound)

    clamp_stats(x0_raw, x0, "start")
    clamp_stats(xf_raw, xf, "goal")

    # ----------------- start/goal validity -----------------
    valid_pairs = []
    for i in range(n_pairs):
        if bool(evc.is_valid_point(x0[i])) and bool(evc.is_valid_point(xf[i])):
            valid_pairs.append((x0[i], xf[i]))

    print(f"[pairs] valid: {len(valid_pairs)}/{n_pairs}")
    if len(valid_pairs) == 0:
        print("[ERROR] No valid start/goal pairs. Reduce ring radius or check voxel collision semantics.")
        sys.exit(1)

    # ----------------- solve -----------------
    total_data = []
    n_ok_quick = 0
    n_ok_full = 0

    for trial, (start_point, goal_point) in enumerate(valid_pairs):
        si, mv_ref, svc_ref = make_space_information(space, evc, emv)

        start = make_state(space, start_point)
        goal = make_state(space, goal_point)

        # -------- Quick solve --------
        ss_quick = og.SimpleSetup(si)
        planner_quick = og.RRTstar(si)
        planner_quick.setRange(planner_range)
        planner_quick.setGoalBias(0.3)
        ss_quick.setPlanner(planner_quick)
        ss_quick.setStartAndGoalStates(start, goal, planner_allowed_termination)
        ss_quick.setup()

        t0 = time.time()
        solved_quick = bool(ss_quick.solve(short_termination_time))
        t_quick = time.time() - t0

        if solved_quick:
            path_quick = path_to_list(ss_quick.getSolutionPath())
            n_ok_quick += 1
        else:
            path_quick = []

        # -------- Full solve --------
        ss_full = og.SimpleSetup(si)
        planner_full = og.RRTstar(si)
        planner_full.setRange(planner_range)
        planner_full.setGoalBias(0.3)
        ss_full.setPlanner(planner_full)
        ss_full.setStartAndGoalStates(start, goal, planner_allowed_termination)
        ss_full.setup()

        t1 = time.time()
        solved_full = bool(ss_full.solve(full_termination_time))
        t_full = time.time() - t1

        if solved_full:
            path_full = path_to_list(ss_full.getSolutionPath())
            n_ok_full += 1
        else:
            path_full = []

        # -------- Post-check validity --------
        st_ok_q, seg_ok_q, bad_i_q = validate_path_with_checkers(path_quick, evc, emv)
        st_ok_f, seg_ok_f, bad_i_f = validate_path_with_checkers(path_full, evc, emv)

        # Path stats
        len_quick = path_length_xyz(path_quick)
        len_full = path_length_xyz(path_full)
        nst_quick = len(path_quick)
        nst_full = len(path_full)

        total_data.append(
            {
                "start": start_point.tolist(),
                "goal": goal_point.tolist(),
                "traj_quick": path_quick,
                "traj_full": path_full,
                "time_quick_s": float(t_quick),
                "time_full_s": float(t_full),
                "n_states_quick": int(nst_quick),
                "n_states_full": int(nst_full),
                "path_len_quick": float(len_quick),
                "path_len_full": float(len_full),
                "postcheck_quick": {
                    "states_valid": bool(st_ok_q),
                    "segments_valid": bool(seg_ok_q),
                    "first_bad_index": int(bad_i_q),
                },
                "postcheck_full": {
                    "states_valid": bool(st_ok_f),
                    "segments_valid": bool(seg_ok_f),
                    "first_bad_index": int(bad_i_f),
                },
            }
        )

        status_q = "OK" if solved_quick else "FAIL"
        status_f = "OK" if solved_full else "FAIL"
        pc_q = "OK" if (st_ok_q and seg_ok_q) else f"BAD@{bad_i_q}"
        pc_f = "OK" if (st_ok_f and seg_ok_f) else f"BAD@{bad_i_f}"

        print(
            f"[trial {trial:03d}] "
            f"quick={status_q} t={t_quick:.3f}s n={nst_quick:3d} L={len_quick:.3f} post={pc_q} | "
            f"full={status_f}  t={t_full:.3f}s n={nst_full:3d} L={len_full:.3f} post={pc_f}"
        )

        del ss_full, planner_full, ss_quick, planner_quick, start, goal, si, mv_ref, svc_ref

    # ----------------- package results -----------------
    stats = {
        "point_checks": int(getattr(evc, "point_count", 0)),
        "line_checks": int(getattr(emv, "line_count", 0)),
        "avg_point_time_ms": (
            1000.0
            * (sum(getattr(evc, "times_point", [])) / max(1, len(getattr(evc, "times_point", []))))
            if hasattr(evc, "times_point")
            else 0.0
        ),
        "avg_line_time_ms": (
            1000.0
            * (sum(getattr(emv, "times_line", [])) / max(1, len(getattr(emv, "times_line", []))))
            if hasattr(emv, "times_line")
            else 0.0
        ),
        "n_trials": int(len(valid_pairs)),
        "n_ok_quick": int(n_ok_quick),
        "n_ok_full": int(n_ok_full),
    }

    data = {
        "scene": scene_name,
        "method": method,
        "vox_npz": vox_npz,
        "robot_radius": float(robot_radius),
        "bounds_from_voxels": True,
        "lower_bound": low_bound.tolist(),
        "upper_bound": high_bound.tolist(),
        "ring": {
            "center": scene_center.tolist(),
            "radius_xy": float(radius_xy),
            "radius_z": float(radius_z),
            "z_freq": float(ring_z_freq),
        },
        "occupancy": {
            "mode": occupancy_mode,
            "tau_rho": float(tau_rho),
            "tau_alpha": float(tau_alpha),
            "max_voxels": int(max_voxels) if max_voxels is not None else None,
        },
        "n_pairs": int(len(valid_pairs)),
        "total_data": total_data,
        "stats": stats,
    }

    os.makedirs("trajs", exist_ok=True)
    out_path = f"trajs/{scene_name}_{method}_svraster_voxels.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
