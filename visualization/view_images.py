from __future__ import annotations
import argparse, re, time, sys
from pathlib import Path

import imageio.v2 as imageio           # pip install imageio
import rerun as rr                     # pip install rerun-sdk
import rerun.blueprint as rrb

'''
python view_images.py     --shutdown /home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/points_init
'''
# ───────────────────────── helpers ────────────────────────────────────────
def load_png(path: Path) -> rr.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return rr.Image(imageio.imread(path))            # H×W×3  uint8


def collect_frame_indices(img_dir: Path, seq_id: str) -> list[int]:
    """
    Return sorted list of frame numbers that have a rendered image.
    Looks for  <seq>_<frame>.jpg
    """
    rex = re.compile(rf"{seq_id}_(\d+)\.jpg")
    frames = sorted(
        int(m.group(1))
        for p in img_dir.glob(f"{seq_id}_*.jpg")
        if (m := rex.match(p.name))
    )
    return frames


# ───────────────────────── main ───────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shutdown", required=True, type=Path,
                    help="…/<seq>_shutdown directory that holds "
                         "'image/' and 'image_gt/' sub-folders")
    ap.add_argument("--autoplay", action="store_true",
                    help="advance frames automatically (2 FPS)")
    rr.script_add_args(ap)
    args = ap.parse_args()

    shut_dir  = args.shutdown
    img_dir   = shut_dir / "image"
    gt_dir    = shut_dir / "image_gt"
    if not img_dir.is_dir() or not gt_dir.is_dir():
        sys.exit("[ERROR] expected 'image/' and 'image_gt/' folders under "
                 f"{shut_dir}")

    # Seq-id is the prefix before first underscore in filenames
    first_file = next(img_dir.glob("*.jpg"), None)
    if first_file is None:
        sys.exit("[ERROR] no rendered images found")
    seq_id = first_file.stem.split("_")[0]

    frames = collect_frame_indices(img_dir, seq_id)
    if not frames:
        sys.exit("[ERROR] could not parse frame indices")

    # ---------- viewer layout -------------------------------------------
    rr.script_setup(
        args,
        f"render_vs_gt_{seq_id}",
        default_blueprint=rrb.Horizontal(
            rrb.Spatial2DView(name="Render", origin="render"),
            rrb.Spatial2DView(name="GT",     origin="gt"),
            column_shares=[1, 1],
        ),
    )

    # ---------- stream ---------------------------------------------------
    for fr in frames:
        rr.set_time(timeline="frame", sequence=fr)

        r_path = img_dir / f"{seq_id}_{fr}.jpg"
        g_path = gt_dir  / f"{seq_id}_{fr}_gt.jpg"

        rr.log("render/img", load_png(r_path))
        if g_path.exists():
            rr.log("gt/img", load_png(g_path))

        if args.autoplay:
            time.sleep(0.5)                         # ≈2 FPS

    rr.script_teardown(args)


if __name__ == "__main__":
    main()