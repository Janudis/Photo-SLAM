from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path

import imageio.v2 as imageio        # pip install imageio
import rerun as rr                  # pip install rerun-sdk
import rerun.blueprint as rrb

'''
python view_images_3way.py   --exp-a ~/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/experiments/model_init_offline_approach   --title-a "Voxel" --title-gt "GT"   --autoplay
'''
def load_png(path: Path) -> rr.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return rr.Image(imageio.imread(path))  # H×W×C uint8


def detect_seq_id(img_dir: Path) -> str:
    """
    Detect '<seq>_<frame>[ _gt].jpg' sequence id.
    Works for plain and '_gt' suffixed names.
    """
    pat = re.compile(r"^(.*?)_\d+(?:_gt)?\.jpg$")
    for p in sorted(img_dir.glob("*.jpg")):
        m = pat.match(p.name)
        if m:
            return m.group(1)
    raise RuntimeError(f"[ERROR] could not detect seq_id in: {img_dir}")


def collect_frame_indices(img_dir: Path, seq_id: str) -> list[int]:
    """
    Return sorted frame indices for '<seq>_<frame>.jpg' or '<seq>_<frame>_gt.jpg'.
    """
    pat = re.compile(rf"^{re.escape(seq_id)}_(\d+)(?:_gt)?\.jpg$")
    frames = []
    for p in img_dir.glob("*.jpg"):
        m = pat.match(p.name)
        if m:
            frames.append(int(m.group(1)))
    return sorted(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="View A (voxel) vs optional B (original) vs GT in Rerun.")
    ap.add_argument("--exp-a", required=True, type=Path,
                    help="Experiment A dir with 'image/' and 'image_gt/' (e.g. model_init_offline_approach)")
    ap.add_argument("--exp-b", type=Path, default=None,
                    help="(Optional) Experiment B dir with 'image/' (e.g. offline_approach_original)")
    ap.add_argument("--title-a", default="Voxel (A)")
    ap.add_argument("--title-b", default="Original (B)")
    ap.add_argument("--title-gt", default="GT")
    ap.add_argument("--autoplay", action="store_true", help="Advance frames automatically (≈2 FPS)")
    rr.script_add_args(ap)
    args = ap.parse_args()

    a_img_dir = args.exp_a / "image"
    gt_dir    = args.exp_a / "image_gt"
    if not a_img_dir.is_dir() or not gt_dir.is_dir():
        sys.exit(f"[ERROR] missing 'image' or 'image_gt' under: {args.exp_a}")

    b_img_dir = None
    if args.exp_b is not None:
        b_img_dir = args.exp_b / "image"
        if not b_img_dir.is_dir():
            sys.exit(f"[ERROR] missing 'image' under: {args.exp_b}")

    # Detect sequence prefixes
    a_seq  = detect_seq_id(a_img_dir)   # e.g. '1901'
    gt_seq = detect_seq_id(gt_dir)      # likely same as a_seq
    b_seq  = detect_seq_id(b_img_dir) if b_img_dir else None  # e.g. '581'

    # Frame sets
    frames_a  = set(collect_frame_indices(a_img_dir, a_seq))
    frames_gt = set(collect_frame_indices(gt_dir,    gt_seq))
    if b_img_dir:
        frames_b = set(collect_frame_indices(b_img_dir, b_seq))
        frames   = sorted(frames_a & frames_b & frames_gt)
    else:
        frames   = sorted(frames_a & frames_gt)

    if not frames:
        sys.exit("[ERROR] No common frames found for the selected streams.")

    # Viewer layout (2 panes if no B; otherwise 3 panes)
    if b_img_dir:
        layout = rrb.Horizontal(
            rrb.Spatial2DView(name=args.title_a,  origin="a"),
            rrb.Spatial2DView(name=args.title_b,  origin="b"),
            rrb.Spatial2DView(name=args.title_gt, origin="gt"),
            column_shares=[1, 1, 1],
        )
        app_title = "A vs B vs GT"
    else:
        layout = rrb.Horizontal(
            rrb.Spatial2DView(name=args.title_a,  origin="a"),
            rrb.Spatial2DView(name=args.title_gt, origin="gt"),
            column_shares=[1, 1],
        )
        app_title = "A vs GT"

    rr.script_setup(args, app_title, default_blueprint=layout)

    # Stream images
    for fr in frames:
        rr.set_time(timeline="frame", sequence=fr)

        # A
        a_path  = a_img_dir / f"{a_seq}_{fr}.jpg"
        rr.log("a/img", load_png(a_path))

        # Optional B
        if b_img_dir:
            b_path = b_img_dir / f"{b_seq}_{fr}.jpg"
            rr.log("b/img", load_png(b_path))

        # GT
        gt_path = gt_dir / f"{gt_seq}_{fr}_gt.jpg"
        rr.log("gt/img", load_png(gt_path))

        if args.autoplay:
            time.sleep(0.5)  # ≈2 FPS

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
