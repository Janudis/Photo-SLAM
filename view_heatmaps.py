# '''
# python view_heatmaps.py   --root /home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/heatmaps   --extrema /home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/extrema   --kf 0   --autoplay
# python view_heatmaps.py   --extrema1 /home/dimitris/Photo-SLAM_original/results/tum_mono_/rgbd_dataset_freiburg1_desk/extrema   --extrema2 /home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/extrema   --kf 0   --autoplay
# '''
# from __future__ import annotations
# import argparse, re, sys, time
# from pathlib import Path

# import imageio.v2 as imageio  # pip install imageio
# import rerun as rr            # pip install rerun-sdk
# import rerun.blueprint as rrb

# # ────────────────────────── Helpers ─────────────────────────────────────────
# def load_png(path: Path) -> rr.Image:
#     if not path.exists():
#         raise FileNotFoundError(path)
#     img = imageio.imread(path)
#     return rr.Image(img)

# # def gather_iters(dir: Path, prefix: str) -> set[int]:
# def gather_iters(dir: Path, file_prefix: str) -> set[int]:
#     """
#     Find all iteration numbers from filenames like
#     …_iter000123…png under `dir`.
#     """
#     # pat = re.compile(rf"{prefix}.*?_iter(\d{{6}}).*?\.png$")
#     pat = re.compile(rf"{file_prefix}_iter(\d{{6}}).*?\.png$")
#     its = set()
#     for p in dir.glob("*.png"):
#         m = pat.match(p.name)
#         if m:
#             its.add(int(m.group(1)))
#     return its

# def pick_any_gt(dir: Path, file_prefix: str, iters: list[int]) -> Path | None:
#     """
#     For a given dir and prefix/kf, find the first existing
#     *_iter######_gt*.png or *_iter######*.png (in that order).
#     """
#     for it in iters:
#         for suf in ("_gt", ""):
#             candidate = dir / f"{file_prefix}_iter{it:06d}{suf}.png"
#             if candidate.exists():
#                 return candidate
#     return None

# # ─────────────────────────── main ────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root",     type=Path,
#                     help="heatmaps parent (contains /kf<i>)")
#     ap.add_argument("--extrema",  type=Path,
#                     help="(optional) single extrema parent (contains /kf<i>)")
#     ap.add_argument("--extrema1", type=Path,
#                     help="first extrema parent (compare mode)")
#     ap.add_argument("--extrema2", type=Path,
#                     help="second extrema parent (compare mode)")
#     ap.add_argument("--kf",       required=True, type=int,
#                     help="key-frame id (integer)")
#     ap.add_argument("--autoplay", action="store_true",
#                     help="advance frames automatically (~2 FPS)")
#     rr.script_add_args(ap)
#     args = ap.parse_args()

#     prefix = f"kf{args.kf}"
#     dir_prefix = f"kf{args.kf}"
#     file_prefix = f"kf{args.kf:04d}"

#     # must have --root in either mode
#     if not args.root:
#         sys.exit("[ERROR] --root is required")

#     # heat_dir = args.root / prefix
#     heat_dir = args.root / dir_prefix
#     if not heat_dir.is_dir():
#         sys.exit(f"[ERROR] heatmaps folder '{heat_dir}' missing")

#     # decide mode
#     compare = bool(args.extrema1 and args.extrema2)
#     if compare:
#         # both extrema1 & extrema2 must exist
#         ext1_dir = (args.extrema1 / prefix) if args.extrema1.name != prefix else args.extrema1
#         ext2_dir = (args.extrema2 / prefix) if args.extrema2.name != prefix else args.extrema2
#         if not ext1_dir.is_dir() or not ext2_dir.is_dir():
#             sys.exit(f"[ERROR] both extrema1/{prefix} and extrema2/{prefix} must exist")
#         print(f"[INFO] compare‐extrema mode:")
#         print(f"   GT from:    {heat_dir}")
#         print(f"   Extrema A:  {ext1_dir}")
#         print(f"   Extrema B:  {ext2_dir}")
#     else:
#         # single‐extrema mode
#         ext_dir = (args.extrema / prefix) if args.extrema and args.extrema.name != prefix else args.extrema or (args.root.parent / "extrema" / prefix)
#         ext_ok = ext_dir and ext_dir.is_dir()
#         print(f"[INFO] heatmap + extrema mode:")
#         print(f"   GT / Heat:  {heat_dir}")
#         print(f"   Extrema:    {ext_dir} ({'OK' if ext_ok else 'missing'})")

#     # collect iterations (union in compare, or just heatmap in single‐extrema)
#     if compare:
#     # extrema‐only mode: filenames like best_masked_iter######.png
#         def gather_iters_any(d: Path) -> set[int]:
#             pat = re.compile(r".*?_iter(\d{6}).*?\.png$")
#             its = set()
#             for p in d.glob("*.png"):
#                 if m := pat.match(p.name):
#                     its.add(int(m.group(1)))
#             return its

#         its1 = gather_iters_any(ext1_dir)
#         its2 = gather_iters_any(ext2_dir)
#         iters = sorted(its1 | its2)
#     else:
#         # heatmap mode: file_prefix_iter######.png
#         iters = sorted(gather_iters(heat_dir, file_prefix))

#     if not iters:
#         sys.exit("[ERROR] no iterations found")

#     # pick GT image (first available _gt or fallback)
#     gt_path = pick_any_gt(heat_dir, file_prefix, iters)
#     if not gt_path:
#         sys.exit("[ERROR] could not locate a ground‐truth (_gt) image")

#     # build blueprint
#     if compare:
#         blueprint = rrb.Horizontal(
#             rrb.Spatial2DView(name="GT",       origin="gt"),
#             rrb.Spatial2DView(name="Extrema A",origin="e1"),
#             rrb.Spatial2DView(name="Extrema B",origin="e2"),
#             column_shares=[1,1,1],
#         )
#     else:
#         blueprint = rrb.Horizontal(
#             rrb.Spatial2DView(name="GT",     origin="gt"),
#             rrb.Spatial2DView(name="HeatMap",origin="hm"),
#             rrb.Spatial2DView(name="Extrema",origin="ext"),
#             column_shares=[1,1,1],
#         )

#     # setup & log GT once
#     rr.script_setup(args, f"voxel_kf{args.kf}", default_blueprint=blueprint)
#     rr.log("gt/img", load_png(gt_path), static=True)

#     # stream
#     for it in iters:
#         rr.set_time(timeline="iter", sequence=it)
#         if compare:            # find _any_ file containing this iteration number
#             p1 = next((f for f in ext1_dir.glob(f"*iter{it:06d}*.png")), None)
#             p2 = next((f for f in ext2_dir.glob(f"*iter{it:06d}*.png")), None)
#             if p1: rr.log("e1/img", load_png(p1))
#             if p2: rr.log("e2/img", load_png(p2))
#         else:
#             # standard heatmap            
#             hm = heat_dir / f"{file_prefix}_iter{it:06d}.png"
#             if hm.exists(): rr.log("hm/img", load_png(hm))

#             # optional extrema            
#             if ext_ok:
#                 ex = next((f for f in ext_dir.glob(f"*iter{it:06d}*.png")), None)
#                 if ex: rr.log("ext/img", load_png(ex))

#         if args.autoplay:
#             time.sleep(0.5)

#     rr.script_teardown(args)


# if __name__ == "__main__":
#     main()


from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path

import imageio.v2 as imageio           # pip install imageio
import rerun as rr                     # pip install rerun-sdk
import rerun.blueprint as rrb

'''
python view_heatmaps.py   --root /home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/heatmaps   --extrema /home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/extrema   --kf 0   --autoplay
'''
# ───────────────────────── helpers ──────────────────────────────────────────
def load_png(path: Path) -> rr.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    img = imageio.imread(path)
    return rr.Image(img)


def find_extrema_file(ext_dir: Path, iter_: int) -> Path | None:
    patt = re.compile(rf".*masked_iter{iter_:06d}_img\d+\.png")
    for p in ext_dir.glob("*masked_iter*.png"):
        if patt.fullmatch(p.name):
            return p
    return None


# ───────────────────────── main ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     required=True, type=Path,
                    help="…/heatmaps directory")
    ap.add_argument("--kf",       required=True, type=int,
                    help="key-frame id (integer)")
    ap.add_argument("--extrema",  type=Path,
                    help="override extrema directory (default: ../extrema/kf<i>)")
    ap.add_argument("--autoplay", action="store_true",
                    help="advance frames automatically (2 FPS)")
    rr.script_add_args(ap)                      # --headless, --connect, …
    args = ap.parse_args()

    kf_id    = args.kf
    prefix   = f"kf{kf_id:04d}"
    heat_dir = args.root / f"kf{kf_id}"
    if not heat_dir.is_dir():
        sys.exit(f"[ERROR] '{heat_dir}' does not exist")

    # ---------- extrema folder ------------------------------------------------
    if args.extrema:
        ext_dir = args.extrema / f"kf{kf_id}"
    else:
        ext_dir = heat_dir.parent.parent / "extrema" / f"kf{kf_id}"
    ext_available = ext_dir.is_dir()

    print(f"[INFO] reading heat-maps from: {heat_dir}")
    if ext_available:
        print(f"[INFO] reading extrema   from: {ext_dir}")
    else:
        print("[WARN] extrema folder not found – panel will stay empty")

    # ---------- collect iterations -------------------------------------------
    rex = re.compile(rf"{prefix}_iter(\d{{6}})(?:_gt)?\.png")
    iters = sorted(
        int(m.group(1))
        for p in heat_dir.glob(f"{prefix}_iter*.png")
        if (m := rex.match(p.name))
    )
    if not iters:
        sys.exit("[ERROR] no heat-map images found")

    gt_path = next(
        (heat_dir / f"{prefix}_iter{it:06d}_gt.png" for it in iters
         if (heat_dir / f"{prefix}_iter{it:06d}_gt.png").exists()),
        heat_dir / f"{prefix}_iter{iters[0]:06d}.png",
    )

    # ---------- viewer layout -------------------------------------------------
    rr.script_setup(
        args,
        f"voxel_heatmaps_kf{kf_id}",
        default_blueprint=rrb.Horizontal(
            rrb.Spatial2DView(name="GT",       origin="gt"),
            # rrb.Spatial2DView(name="HeatMap",  origin="hm"),
            rrb.Spatial2DView(name="Extrema",  origin="ext"),
            column_shares=[1, 1, 1],
        ),
    )

    rr.log("gt/img", load_png(gt_path), static=True)           # GT never changes

    # ---------- stream --------------------------------------------------------
    for it in iters:
        rr.set_time(timeline="iter", sequence=it)

        hm_path = heat_dir / f"{prefix}_iter{it:06d}.png"
        if hm_path.exists():
            rr.log("hm/img", load_png(hm_path))

        if ext_available:
            ext_path = find_extrema_file(ext_dir, it)
            if ext_path is not None:
                rr.log("ext/img", load_png(ext_path))

        if args.autoplay:
            time.sleep(0.5)                                   # ~2 FPS

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
