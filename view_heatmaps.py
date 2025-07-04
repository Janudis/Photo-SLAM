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
            rrb.Spatial2DView(name="HeatMap",  origin="hm"),
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
