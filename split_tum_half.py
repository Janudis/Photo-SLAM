# #!/usr/bin/env python3
# import argparse, os, shutil, sys
# from pathlib import Path

# def read_tum_list(file_path):
#     """
#     Returns:
#       header_lines: list[str]     (#-comment lines)
#       rows: list[tuple[float,str]]  (timestamp, rel_path) for rgb/depth
#     For groundtruth/accelerometer we’ll re-use this but only timestamp is used.
#     """
#     header, rows = [], []
#     with open(file_path, "r") as f:
#         for ln in f:
#             ln = ln.strip()
#             if not ln:
#                 continue
#             if ln.startswith("#"):
#                 header.append(ln)
#                 continue
#             parts = ln.split()
#             if len(parts) == 0:
#                 continue
#             try:
#                 ts = float(parts[0])
#             except ValueError:
#                 # Treat unparseable line as comment
#                 header.append("# " + ln)
#                 continue
#             if len(parts) >= 2:
#                 rows.append((ts, parts[1]))
#             else:
#                 # for streams with only timestamp (rare) – store empty path
#                 rows.append((ts, ""))
#     return header, rows

# def write_tum_list(file_path, header, rows):
#     file_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(file_path, "w") as f:
#         for h in header:
#             f.write(h + "\n")
#         for ts, rel in rows:
#             if rel:
#                 f.write(f"{ts:.6f} {rel}\n")
#             else:
#                 f.write(f"{ts:.6f}\n")

# def read_tum_scalar_stream(file_path):
#     """For groundtruth.txt/accelerometer.txt where extra columns exist.
#        Keeps the whole line but filters by timestamp."""
#     header, lines = [], []
#     with open(file_path, "r") as f:
#         for ln in f:
#             s = ln.strip()
#             if not s:
#                 continue
#             if s.startswith("#"):
#                 header.append(s)
#                 continue
#             parts = s.split()
#             try:
#                 ts = float(parts[0])
#             except ValueError:
#                 header.append("# " + s)
#                 continue
#             lines.append((ts, s))
#     return header, lines

# def write_tum_scalar_stream(file_path, header, lines):
#     file_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(file_path, "w") as f:
#         for h in header:
#             f.write(h + "\n")
#         for _, s in lines:
#             f.write(s + "\n")

# def copy_or_link(src, dst, symlink):
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if symlink:
#         # try to create a relative symlink; fallback to copy on failure
#         try:
#             rel = os.path.relpath(src, start=dst.parent)
#             if dst.exists() or dst.is_symlink():
#                 dst.unlink()
#             os.symlink(rel, dst)
#             return
#         except Exception:
#             pass
#     # fallback: copy
#     shutil.copy2(src, dst)

# def main():
#     ap = argparse.ArgumentParser(description="Split TUM RGB-D dataset to first half (by time)")
#     ap.add_argument("--in_dir",  required=True, help="Path to original TUM dataset folder")
#     ap.add_argument("--out_dir", required=True, help="Path to output (first-half) dataset folder")
#     ap.add_argument("--symlink", action="store_true", help="Use symlinks for images instead of copying")
#     ap.add_argument("--by", choices=["time","count"], default="time",
#                     help="Half by time (default) or by number of rows in rgb.txt")
#     args = ap.parse_args()

#     in_dir  = Path(args.in_dir).resolve()
#     out_dir = Path(args.out_dir).resolve()
#     if out_dir.exists():
#         print(f"[INFO] Removing existing out_dir: {out_dir}")
#         shutil.rmtree(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Required files
#     rgb_txt  = in_dir/"rgb.txt"
#     depth_txt= in_dir/"depth.txt"
#     gt_txt   = in_dir/"groundtruth.txt"
#     acc_txt  = in_dir/"accelerometer.txt"
#     cam_yaml = in_dir/"camera.yaml"

#     # read rgb/depth lists
#     rgb_header,  rgb_rows  = read_tum_list(rgb_txt)
#     depth_header,depth_rows= read_tum_list(depth_txt)

#     if len(rgb_rows) == 0:
#         print("[ERR] rgb.txt has no valid rows.", file=sys.stderr)
#         sys.exit(1)

#     # decide split threshold
#     if args.by == "time":
#         t0 = rgb_rows[0][0]
#         t1 = rgb_rows[-1][0]
#         t_mid = t0 + 0.5*(t1 - t0)
#         rgb_keep = [(t, p) for (t, p) in rgb_rows  if t <= t_mid]
#         depth_keep = [(t, p) for (t, p) in depth_rows if t <= t_mid]
#         print(f"[INFO] Time split: t0={t0:.6f}  t1={t1:.6f}  t_mid={t_mid:.6f}")
#     else:
#         # half by count using rgb as reference
#         half_n = max(1, len(rgb_rows)//2)
#         t_mid = rgb_rows[half_n-1][0]
#         rgb_keep = rgb_rows[:half_n]
#         # keep all depth rows with timestamp <= corresponding t_mid
#         depth_keep = [(t, p) for (t, p) in depth_rows if t <= t_mid]
#         print(f"[INFO] Count split: total rgb={len(rgb_rows)}  keep={len(rgb_keep)}  t_mid={t_mid:.6f}")

#     print(f"[INFO] Keep rgb:   {len(rgb_keep)} / {len(rgb_rows)}")
#     print(f"[INFO] Keep depth: {len(depth_keep)} / {len(depth_rows)}")

#     # filter groundtruth & accelerometer by the same time window [min_ts, t_mid]
#     min_ts = rgb_keep[0][0] if rgb_keep else rgb_rows[0][0]
#     gt_header,  gt_lines  = read_tum_scalar_stream(gt_txt)   if gt_txt.exists() else ([], [])
#     acc_header, acc_lines = read_tum_scalar_stream(acc_txt)  if acc_txt.exists() else ([], [])
#     gt_keep  = [(t, s) for (t, s) in gt_lines  if (t >= min_ts and t <= t_mid)]
#     acc_keep = [(t, s) for (t, s) in acc_lines if (t >= min_ts and t <= t_mid)]
#     print(f"[INFO] Keep groundtruth:     {len(gt_keep)} / {len(gt_lines)}")
#     print(f"[INFO] Keep accelerometer:   {len(acc_keep)} / {len(acc_lines)}")

#     # write filtered text files
#     write_tum_list(out_dir/"rgb.txt",   rgb_header,   rgb_keep)
#     write_tum_list(out_dir/"depth.txt", depth_header, depth_keep)
#     if gt_lines:
#         write_tum_scalar_stream(out_dir/"groundtruth.txt", gt_header, gt_keep)
#     if acc_lines:
#         write_tum_scalar_stream(out_dir/"accelerometer.txt", acc_header, acc_keep)

#     # copy camera.yaml
#     if cam_yaml.exists():
#         shutil.copy2(cam_yaml, out_dir/"camera.yaml")

#     # copy/symlink images that are referenced
#     # (preserve folder structure "rgb/<file>" and "depth/<file>")
#     copied = 0
#     for _, rel in rgb_keep:
#         src = in_dir/rel
#         dst = out_dir/rel
#         if not src.exists():
#             print(f"[WARN] missing RGB file: {src}")
#             continue
#         copy_or_link(src, dst, args.symlink)
#         copied += 1
#     print(f"[INFO] RGB files placed: {copied}")

#     copied = 0
#     for _, rel in depth_keep:
#         src = in_dir/rel
#         dst = out_dir/rel
#         if not src.exists():
#             print(f"[WARN] missing depth file: {src}")
#             continue
#         copy_or_link(src, dst, args.symlink)
#         copied += 1
#     print(f"[INFO] Depth files placed: {copied}")

#     print("\n[OK] First-half dataset ready:")
#     print(f"     {out_dir}")
#     print("     You can now point Photo-SLAM to this folder to build a smaller scene AABB.")

# if __name__ == "__main__":
#     main()

# shrink_tum_dataset.py
import os, math, random

# === CONFIG ===
DATASET = "/home/dimitris/Photo-SLAM/scripts/data/rgbd_dataset_freiburg1_desk"
OUT     = DATASET + "_third"
FRACTION = 1/3            # set to e.g. 0.33
MODE     = "uniform"      # "uniform" or "head"
RNG_SEED = 0              # for reproducibility when MODE="uniform"

# ============== helpers ==============
def read_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    comments = [l for l in lines if l.lstrip().startswith("#")]
    data     = [l for l in lines if not l.lstrip().startswith("#") and l.strip()]
    return comments, data

def select_indices(n, k, mode="uniform"):
    if k <= 0: return []
    if k >= n: return list(range(n))
    if mode == "head":
        return list(range(k))
    # uniform
    random.seed(RNG_SEED)
    # even spread with slight jitter; preserves order
    idxs = [round(i*(n-1)/(k-1)) for i in range(k)] if k > 1 else [0]
    # make unique & sorted
    idxs = sorted(set(idxs))
    # if dedup shrunk, pad with random unused indices
    while len(idxs) < k:
        cand = random.randrange(n)
        if cand not in idxs:
            idxs.append(cand)
    return sorted(idxs)

def write_and_copy(img_list, comments, in_root, out_root, subdir, out_txt):
    os.makedirs(os.path.join(out_root, subdir), exist_ok=True)

    with open(out_txt, "w") as f:
        f.writelines(comments)
        for line in img_list:
            f.write(line)

    # copy files referenced in txt
    copied, missing = 0, 0
    for line in img_list:
        parts = line.strip().split()
        if len(parts) < 2: 
            continue
        rel = parts[1]                # e.g. "rgb/xxxxx.png" or "depth/xxxxx.png"
        src = os.path.join(in_root, rel)
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            os.system(f'cp "{src}" "{dst}"')
            copied += 1
        else:
            missing += 1
    return copied, missing

# ============== main ==============
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "rgb"), exist_ok=True)
os.makedirs(os.path.join(OUT, "depth"), exist_ok=True)

# rgb.txt
rgb_c, rgb_d = read_txt(os.path.join(DATASET, "rgb.txt"))
k_rgb = max(1, math.floor(len(rgb_d) * FRACTION))
keep_idx_rgb = select_indices(len(rgb_d), k_rgb, MODE)
rgb_kept = [rgb_d[i] for i in keep_idx_rgb]

# depth.txt
dep_c, dep_d = read_txt(os.path.join(DATASET, "depth.txt"))
k_dep = max(1, math.floor(len(dep_d) * FRACTION))
keep_idx_dep = select_indices(len(dep_d), k_dep, MODE)
dep_kept = [dep_d[i] for i in keep_idx_dep]

# write + copy
rgb_copied, rgb_missing = write_and_copy(
    rgb_kept, rgb_c, DATASET, OUT, "rgb", os.path.join(OUT, "rgb.txt")
)
dep_copied, dep_missing = write_and_copy(
    dep_kept, dep_c, DATASET, OUT, "depth", os.path.join(OUT, "depth.txt")
)

# copy metadata (unchanged)
for meta in ["camera.yaml", "groundtruth.txt", "accelerometer.txt"]:
    src = os.path.join(DATASET, meta)
    dst = os.path.join(OUT, meta)
    if os.path.exists(src):
        os.system(f'cp "{src}" "{dst}"')

print("=== DONE ===")
print(f"rgb:   kept {len(rgb_kept)} / {len(rgb_d)} entries | copied {rgb_copied}, missing files {rgb_missing}")
print(f"depth: kept {len(dep_kept)} / {len(dep_d)} entries | copied {dep_copied}, missing files {dep_missing}")
print(f"Output: {OUT}")
print(f"Mode: {MODE}, Fraction: {FRACTION}")

