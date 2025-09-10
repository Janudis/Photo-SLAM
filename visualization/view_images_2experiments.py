# view_images_2experiments.py
'''
python view_images_2experiments.py   --dir-a ~/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/experiments/model_init_offline_approach_densif/image   --dir-b ~/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/experiments/model_init_offline_with_aggresive_densif/image   --title-a "Densif" --title-b "Aggressive Densif"   --autoplay
'''
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path

import imageio.v2 as imageio
import rerun as rr
import rerun.blueprint as rrb


def load_img(path: Path) -> rr.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return rr.Image(imageio.imread(path))  # H×W×C uint8


def detect_seq_id(img_dir: Path) -> str:
    """
    Detect '<seq>_<frame>.jpg' sequence id in a directory.
    """
    pat = re.compile(r"^(.*?)_\d+\.jpg$")
    for p in sorted(img_dir.glob("*.jpg")):
        m = pat.match(p.name)
        if m:
            return m.group(1)
    raise RuntimeError(f"[ERROR] could not detect seq_id in: {img_dir}")


def collect_frame_indices(img_dir: Path, seq_id: str) -> list[int]:
    """
    Return sorted frame indices for '<seq>_<frame>.jpg'.
    """
    pat = re.compile(rf"^{re.escape(seq_id)}_(\d+)\.jpg$")
    frames = []
    for p in img_dir.glob("*.jpg"):
        m = pat.match(p.name)
        if m:
            frames.append(int(m.group(1)))
    return sorted(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two experiments (A vs B) side-by-side in Rerun.")
    ap.add_argument("--dir-a", required=True, type=Path, help="Directory with JPGs for experiment A")
    ap.add_argument("--dir-b", required=True, type=Path, help="Directory with JPGs for experiment B")
    ap.add_argument("--title-a", default="Experiment A")
    ap.add_argument("--title-b", default="Experiment B")
    ap.add_argument("--autoplay", action="store_true", help="Advance frames automatically (≈2 FPS)")
    rr.script_add_args(ap)
    args = ap.parse_args()

    for d in [args.dir_a, args.dir_b]:
        if not d.is_dir():
            sys.exit(f"[ERROR] not a directory: {d}")

    # Detect per-dir sequence prefixes
    a_seq = detect_seq_id(args.dir_a)   # e.g. '1901'
    b_seq = detect_seq_id(args.dir_b)   # e.g. '781'

    # Gather frames & align on intersection
    frames_a = set(collect_frame_indices(args.dir_a, a_seq))
    frames_b = set(collect_frame_indices(args.dir_b, b_seq))
    frames   = sorted(frames_a & frames_b)
    if not frames:
        sys.exit("[ERROR] No common frames between A and B.")

    # Viewer layout: two panes
    rr.script_setup(
        args,
        "A_vs_B",
        default_blueprint=rrb.Horizontal(
            rrb.Spatial2DView(name=args.title_a, origin="a"),
            rrb.Spatial2DView(name=args.title_b, origin="b"),
            column_shares=[1, 1],
        ),
    )

    # Stream images
    for fr in frames:
        rr.set_time(timeline="frame", sequence=fr)

        a_path = args.dir_a / f"{a_seq}_{fr}.jpg"
        b_path = args.dir_b / f"{b_seq}_{fr}.jpg"

        rr.log("a/img", load_img(a_path))
        rr.log("b/img", load_img(b_path))

        if args.autoplay:
            time.sleep(0.5)  # ≈2 FPS

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
