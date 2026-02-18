import json
import re
import numpy as np


# ----------------------------
# Loading utilities
# ----------------------------
def load_slam_centers_tum(path):
    """
    TUM format: t tx ty tz qx qy qz qw
    Returns:
      times: (N,) float64
      centers: (N,3) float64
    """
    ts = []
    Cs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            t = float(vals[0])
            tx, ty, tz = map(float, vals[1:4])
            ts.append(t)
            Cs.append([tx, ty, tz])
    if len(ts) == 0:
        raise RuntimeError(f"No SLAM poses parsed from: {path}")
    return np.asarray(ts, dtype=np.float64), np.asarray(Cs, dtype=np.float64)


def load_ns_centers_transforms(path, fps_fallback=30.0):
    """
    Nerfstudio transforms.json: frames[*].transform_matrix is c2w.
    Prefer frames[*].timestamp if available.
    Else fall back to parsing frame_(\\d+).png and converting to time = idx/fps_fallback.

    Returns:
      times: (M,) float64
      centers: (M,3) float64
      frame_ids: (M,) int64  (parsed frame index if available, else -1)
      mode: "timestamp" or "frameidx"
    """
    with open(path, "r") as f:
        js = json.load(f)

    pat = re.compile(r"frame_(\d+)\.png")

    times = []
    centers = []
    frame_ids = []

    has_timestamp = False
    for fr in js.get("frames", []):
        if "timestamp" in fr:
            has_timestamp = True
            break

    mode = "timestamp" if has_timestamp else "frameidx"

    for fr in js.get("frames", []):
        T = np.array(fr["transform_matrix"], dtype=np.float64)
        c = T[:3, 3].copy()

        if mode == "timestamp":
            t = float(fr["timestamp"])
            fid = -1
            # still try parse frame id for logging
            m = pat.search(fr.get("file_path", ""))
            if m:
                fid = int(m.group(1))
        else:
            m = pat.search(fr.get("file_path", ""))
            if not m:
                continue
            fid = int(m.group(1))
            t = fid / float(fps_fallback)

        times.append(t)
        centers.append(c)
        frame_ids.append(fid)

    if len(times) == 0:
        raise RuntimeError(f"No NS frames parsed from: {path} (mode={mode})")

    times = np.asarray(times, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)

    # sort by time (important for matching)
    order = np.argsort(times)
    return times[order], centers[order], frame_ids[order], mode


# ----------------------------
# Matching
# ----------------------------
def match_by_nearest_time(ts_slam, C_slam, ts_ns, C_ns, max_dt):
    """
    Nearest-neighbor in time (1D).
    Returns matched arrays:
      src (K,3) from SLAM
      dst (K,3) from NS
      dt  (K,)  time offsets
      idx_slam, idx_ns indices of kept matches
    """
    # ensure NS times sorted
    ns_order = np.argsort(ts_ns)
    ts_ns = ts_ns[ns_order]
    C_ns = C_ns[ns_order]

    idx_ns_list = []
    idx_slam_list = []
    dt_list = []

    for i, t in enumerate(ts_slam):
        j = np.searchsorted(ts_ns, t)
        cand = []
        if j - 1 >= 0:
            cand.append(j - 1)
        if j < len(ts_ns):
            cand.append(j)

        if not cand:
            continue

        # pick nearest
        cand = np.array(cand, dtype=np.int64)
        dts = np.abs(ts_ns[cand] - t)
        best = cand[np.argmin(dts)]
        dt = ts_ns[best] - t

        if np.abs(dt) <= max_dt:
            idx_slam_list.append(i)
            idx_ns_list.append(best)
            dt_list.append(dt)

    idx_slam = np.asarray(idx_slam_list, dtype=np.int64)
    idx_ns = np.asarray(idx_ns_list, dtype=np.int64)
    dt = np.asarray(dt_list, dtype=np.float64)

    src = C_slam[idx_slam]
    dst = C_ns[idx_ns]
    return src, dst, dt, idx_slam, idx_ns


# ----------------------------
# Umeyama Sim(3): dst ~= s * src * R^T + t
# ----------------------------
def umeyama_sim3(src, dst, with_scale=True):
    """
    Solve: dst ~= s * (src @ R.T) + t
    src,dst: (N,3) row vectors
    Returns: s (float), R (3,3), t (3,)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / n  # (3,3)

    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt

    # enforce proper rotation
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    if with_scale:
        var_src = (X**2).sum() / n
        s = (S.sum()) / var_src
    else:
        s = 1.0

    t = mu_dst - s * (mu_src @ R.T)
    return float(s), R, t


def sim3_apply_points(P, s, R, t):
    return (s * (P @ R.T)) + t.reshape(1, 3)


def robust_fit_sim3(src, dst, trim_frac=0.10, iters=3, with_scale=True):
    """
    Iteratively:
      fit Sim3
      compute residuals
      drop top trim_frac residuals
    """
    assert 0.0 <= trim_frac < 0.5
    keep = np.ones((src.shape[0],), dtype=bool)

    s = 1.0
    R = np.eye(3)
    t = np.zeros((3,))

    for k in range(iters):
        s, R, t = umeyama_sim3(src[keep], dst[keep], with_scale=with_scale)
        pred = sim3_apply_points(src, s, R, t)
        res = np.linalg.norm(pred - dst, axis=1)

        # compute new keep mask
        if trim_frac > 0.0:
            thr = np.quantile(res[keep], 1.0 - trim_frac)
            keep = res <= thr

    # final stats
    pred = sim3_apply_points(src, s, R, t)
    res = np.linalg.norm(pred - dst, axis=1)
    return s, R, t, res, keep


def summarize_cloud(tag, C):
    med = np.median(C, axis=0)
    p1 = np.percentile(C, 1, axis=0)
    p99 = np.percentile(C, 99, axis=0)
    print(f"[{tag}] median={med}, p1={p1}, p99={p99}, span={p99-p1}")


# ----------------------------
# Main
# ----------------------------
def main():
    slam_path = "/home/dimitris/Photo-SLAM/results/statues_voxel/KeyFrameTrajectory_TUM.txt"
    ns_path   = "/home/dimitris/Photo-SLAM/scripts/data/statues/transforms.json"

    # If transforms.json has timestamps, this is only a fallback.
    fps_fallback = 30.0

    # Matching tolerance (seconds).
    # If using NS timestamps: set to something realistic (e.g. 0.02–0.05).
    # If using frameidx fallback: should be <= 0.5/fps.
    max_dt = 0.05

    # Robust trimming
    trim_frac = 0.10   # drop top 10% residuals each iteration
    trim_iters = 4

    ts_slam, C_slam = load_slam_centers_tum(slam_path)
    ts_ns, C_ns, frame_ids, mode = load_ns_centers_transforms(ns_path, fps_fallback=fps_fallback)

    print(f"[info] SLAM poses: {len(ts_slam)}")
    print(f"[info] NS frames : {len(ts_ns)} (mode={mode})")

    # Basic sanity summaries (raw, before fitting)
    summarize_cloud("slam_raw", C_slam)
    summarize_cloud("ns_raw", C_ns)

    # Match by nearest timestamp
    src, dst, dt, idx_slam, idx_ns = match_by_nearest_time(
        ts_slam, C_slam, ts_ns, C_ns, max_dt=max_dt
    )

    print(f"[match] kept: {len(src)} matches (max_dt={max_dt}s)")
    if len(src) < 30:
        raise RuntimeError(
            f"Too few correspondences ({len(src)}). "
            f"Likely timestamp mismatch. Inspect transforms.json for 'timestamp' or adjust max_dt/fps_fallback."
        )

    print(f"[match] dt stats: mean={dt.mean():.6f}s, std={dt.std():.6f}s, maxabs={np.max(np.abs(dt)):.6f}s")

    # Robust Sim3 fit
    s, R, t, res, keep = robust_fit_sim3(
        src, dst, trim_frac=trim_frac, iters=trim_iters, with_scale=True
    )

    rmse_all = float(np.sqrt(np.mean(res**2)))
    rmse_in  = float(np.sqrt(np.mean(res[keep]**2)))
    print(f"[sim3] s={s:.6f}")
    print(f"[sim3] rmse_all={rmse_all:.6f}, rmse_inliers={rmse_in:.6f}, kept_inliers={int(keep.sum())}/{len(keep)}")

    # Diagnostics: after-transform SLAM camera centers should look like NS camera centers
    pred = sim3_apply_points(src, s, R, t)
    summarize_cloud("pred_slam_to_ns", pred)
    summarize_cloud("dst_ns_matched", dst)

    out = {
        "scene": "statues",
        "slam_path": slam_path,
        "ns_path": ns_path,
        "ns_mode": mode,
        "fps_fallback": float(fps_fallback),
        "max_dt_s": float(max_dt),
        "n_corr_raw": int(len(src)),
        "trim_frac": float(trim_frac),
        "trim_iters": int(trim_iters),
        "n_inliers": int(keep.sum()),
        "s": float(s),
        "R": R.tolist(),
        "t": t.tolist(),
        "rmse_all": rmse_all,
        "rmse_inliers": rmse_in,
        "residual_percentiles_m": {
            "p50": float(np.percentile(res, 50)),
            "p90": float(np.percentile(res, 90)),
            "p95": float(np.percentile(res, 95)),
            "p99": float(np.percentile(res, 99)),
            "max": float(np.max(res)),
        },
    }

    out_path = "sim3_slam_to_ns.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
