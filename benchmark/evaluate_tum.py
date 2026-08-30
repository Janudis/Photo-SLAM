#!/usr/bin/env python3
"""Collect TUM appearance metrics for Ours, Ours+MVS, and HI-SLAM2."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import evaluate as common


TUM_SEQUENCES = (
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg2_xyz",
    "rgbd_dataset_freiburg3_long_office_household",
)
VOXEL_METHODS = ("ours", "ours_mvs")
METHOD_LABELS = {
    "ours": "Ours",
    "ours_mvs": "Ours+MVS",
    "hislam2": "HI-SLAM2",
}
METRIC_LABELS = {
    "psnr": ("PSNR", "\\uparrow"),
    "ssim": ("SSIM", "\\uparrow"),
    "lpips": ("LPIPS", "\\downarrow"),
}


class TumEvaluationError(RuntimeError):
    pass


def finite_float(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TumEvaluationError(f"Invalid {description}: {value!r}") from error
    if not math.isfinite(result):
        raise TumEvaluationError(f"Non-finite {description}: {value!r}")
    return result


def resolve_job_path(
    raw_path: str | None,
    fallback: Path,
    description: str,
) -> Path:
    candidates = []
    if raw_path:
        candidates.append(Path(raw_path))
    candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise TumEvaluationError(f"Missing {description}; tried {rendered}")


def load_voxel_jobs(run_root: Path) -> dict[tuple[str, str], common.Job]:
    manifest = common.load_json(run_root / "run_manifest.json", "TUM run manifest")
    if manifest.get("status") != "complete":
        raise TumEvaluationError(
            f"Voxel benchmark is not complete: {manifest.get('status')!r}"
        )
    if manifest.get("dataset") != "tum":
        raise TumEvaluationError("Voxel benchmark is not a TUM run")

    jobs: dict[tuple[str, str], common.Job] = {}
    for raw in manifest.get("jobs", []):
        method = str(raw.get("method", ""))
        sequence = str(raw.get("sequence", ""))
        trial = int(raw.get("trial", 0))
        if method not in VOXEL_METHODS or sequence not in TUM_SEQUENCES:
            continue
        if raw.get("status") != "complete" or trial != 1:
            raise TumEvaluationError(f"Incomplete TUM voxel job: {raw}")
        key = (sequence, method)
        if key in jobs:
            raise TumEvaluationError(f"Duplicate TUM voxel job: {key}")
        fallback_root = (
            run_root / "tum" / sequence / method / "trial_01"
        )
        raw_shutdown = str(raw.get("shutdown_dir", "")) or None
        shutdown_name = Path(raw_shutdown).name if raw_shutdown else ""
        fallback = fallback_root / shutdown_name if shutdown_name else fallback_root
        shutdown = resolve_job_path(raw_shutdown, fallback, "shutdown directory")
        if shutdown == fallback_root.resolve():
            shutdown_candidates = sorted(fallback_root.glob("*_shutdown"))
            if len(shutdown_candidates) != 1:
                raise TumEvaluationError(
                    f"Expected one shutdown directory below {fallback_root}"
                )
            shutdown = shutdown_candidates[0].resolve()
        jobs[key] = common.Job(method, sequence, trial, shutdown)

    expected = {
        (sequence, method)
        for sequence in TUM_SEQUENCES
        for method in VOXEL_METHODS
    }
    if set(jobs) != expected:
        raise TumEvaluationError(
            f"Incomplete TUM voxel matrix; missing={sorted(expected - set(jobs))}"
        )
    return jobs


def load_hislam2_rows(run_root: Path) -> dict[str, dict[str, Any]]:
    manifest = common.load_json(
        run_root / "run_manifest.json", "HI-SLAM2 TUM run manifest"
    )
    if manifest.get("status") != "complete":
        raise TumEvaluationError(
            f"HI-SLAM2 benchmark is not complete: {manifest.get('status')!r}"
        )
    if manifest.get("dataset") != "tum" or manifest.get("method") != "hislam2":
        raise TumEvaluationError("Baseline manifest is not a HI-SLAM2 TUM run")

    rows: dict[str, dict[str, Any]] = {}
    for raw in manifest.get("jobs", []):
        sequence = str(raw.get("sequence", ""))
        if sequence not in TUM_SEQUENCES:
            continue
        if raw.get("status") != "complete" or int(raw.get("trial", 0)) != 1:
            raise TumEvaluationError(f"Incomplete HI-SLAM2 TUM job: {raw}")
        if sequence in rows:
            raise TumEvaluationError(f"Duplicate HI-SLAM2 TUM job: {sequence}")
        fallback = (
            run_root
            / "tum"
            / sequence
            / "hislam2"
            / "trial_01"
            / "native"
            / "psnr"
            / "after_opt"
            / "final_result.json"
        )
        metrics_path = resolve_job_path(
            str(raw.get("appearance_metrics", "")) or None,
            fallback,
            "HI-SLAM2 appearance summary",
        )
        metrics = common.load_json(metrics_path, "HI-SLAM2 appearance summary")
        render_dir = common.require_directory(
            metrics_path.parents[2] / "renders" / "image_after_opt",
            "HI-SLAM2 rendered images",
        )
        evaluated_frames = len(
            [
                path
                for path in render_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in common.IMAGE_SUFFIXES
            ]
        )
        if evaluated_frames == 0:
            raise TumEvaluationError(f"No HI-SLAM2 rendered images in {render_dir}")
        rows[sequence] = {
            "evaluated_frames": evaluated_frames,
            "psnr": finite_float(metrics.get("mean_psnr"), "HI-SLAM2 PSNR"),
            "ssim": finite_float(metrics.get("mean_ssim"), "HI-SLAM2 SSIM"),
            "lpips": finite_float(metrics.get("mean_lpips"), "HI-SLAM2 LPIPS"),
            "metrics_path": str(metrics_path),
        }

    missing = sorted(set(TUM_SEQUENCES) - set(rows))
    if missing:
        raise TumEvaluationError(f"Missing HI-SLAM2 TUM jobs: {missing}")
    return rows


def tum_images(data_root: Path, sequence: str) -> list[Path]:
    sequence_dir = data_root / "TUM" / sequence
    rgb_list = common.require_file(sequence_dir / "rgb.txt", "TUM RGB list")
    images: list[Path] = []
    for line_number, raw in enumerate(
        rgb_list.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise TumEvaluationError(
                f"Invalid TUM RGB row at {rgb_list}:{line_number}"
            )
        image = sequence_dir / fields[1]
        common.require_file(image, "TUM RGB image")
        images.append(image)
    if not images:
        raise TumEvaluationError(f"No RGB images listed by {rgb_list}")
    return images


def average_rows(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        metric: common.mean(float(rows[sequence][metric]) for sequence in TUM_SEQUENCES)
        for metric in ("psnr", "ssim", "lpips")
    }


def ground_truth_image_index(job: common.Job) -> dict[int, Path]:
    image_dir = common.require_directory(
        job.shutdown_dir / "image_gt", "saved ground-truth images"
    )
    result: dict[int, Path] = {}
    pattern = re.compile(r"^\d+_(\d+)(?:_shutdown)?_gt$")
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in common.IMAGE_SUFFIXES:
            continue
        match = pattern.fullmatch(path.stem)
        if not match:
            continue
        keyframe = int(match.group(1))
        if keyframe in result:
            raise TumEvaluationError(
                f"Duplicate ground-truth image for keyframe {keyframe}: {path}"
            )
        result[keyframe] = path
    if not result:
        raise TumEvaluationError(f"No saved ground-truth images in {image_dir}")
    return result


def lpips_for_voxel_frames(
    job: common.Job,
    metrics: dict[int, dict[str, Any]],
    frame_ids: list[int],
    device: str,
    cache_path: Path,
    runtime: tuple[Any, Any, Any],
) -> dict[int, float]:
    if cache_path.is_file():
        cached = common.load_json(cache_path, "TUM LPIPS cache")
        values = {int(key): float(value) for key, value in cached.items()}
        if set(values) == set(frame_ids):
            return values

    torch, Image, model = runtime
    rendered = common.rendered_image_index(job)
    ground_truth = ground_truth_image_index(job)
    values: dict[int, float] = {}
    with torch.inference_mode():
        for index, frame in enumerate(frame_ids, start=1):
            keyframe = int(metrics[frame]["keyframe_id"])
            if keyframe not in rendered or keyframe not in ground_truth:
                raise TumEvaluationError(
                    f"Missing render/ground truth for keyframe {keyframe} in "
                    f"{job.shutdown_dir}"
                )
            render_tensor = common.image_tensor(
                torch, Image, rendered[keyframe], None
            ).to(device)
            size = tuple(Image.open(rendered[keyframe]).size)
            gt_tensor = common.image_tensor(
                torch, Image, ground_truth[keyframe], size
            ).to(device)
            values[frame] = float(model(render_tensor, gt_tensor).item())
            if index % 25 == 0 or index == len(frame_ids):
                print(
                    f"[LPIPS] {job.sequence}/{job.method}: "
                    f"{index}/{len(frame_ids)}",
                    flush=True,
                )
    common.write_json(cache_path, {str(key): value for key, value in values.items()})
    return values


def latex_rows(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for method in ("hislam2", "ours", "ours_mvs"):
        label = METHOD_LABELS[method]
        rows = summary["sequences"]
        average = summary["average"][method]
        for index, metric in enumerate(("psnr", "ssim", "lpips")):
            metric_label, arrow = METRIC_LABELS[metric]
            prefix = f"\\multirow{{3}}{{*}}{{{label}}}" if index == 0 else ""
            values = [float(rows[sequence][method][metric]) for sequence in TUM_SEQUENCES]
            precision = 3
            formatted = " & ".join(f"{value:.{precision}f}" for value in values)
            lines.append(
                f"{prefix}\n& {metric_label} ${arrow}$ & {formatted} & "
                f"{float(average[metric]):.{precision}f} \\\\"
            )
        if method != "ours_mvs":
            lines.append("\\midrule")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-run-id", required=True)
    parser.add_argument("--hislam2-run-id", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("PHOTOSLAM_DATA_ROOT", "/datasets")),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("PHOTOSLAM_RESULTS_ROOT", "/results")),
    )
    parser.add_argument("--lpips-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_root = args.results_root / "paper_benchmark"
    ours_root = benchmark_root / args.ours_run_id
    hislam2_root = benchmark_root / args.hislam2_run_id
    output = args.output or ours_root / "evaluation" / "tum_summary.json"
    voxel_jobs = load_voxel_jobs(ours_root)
    hislam2_rows = load_hislam2_rows(hislam2_root)
    runtime = None if args.skip_lpips else common.load_lpips(args.lpips_device)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "ours_run_id": args.ours_run_id,
        "hislam2_run_id": args.hislam2_run_id,
        "protocol": {
            "ours_frame_set": (
                "intersection of Ours and Ours+MVS mapper keyframes for each sequence"
            ),
            "ours_psnr_ssim": "native mapper metrics before JPEG export",
            "ours_lpips": (
                "AlexNet LPIPS against the mapper's saved undistorted "
                "ground-truth image"
            ),
            "hislam2": (
                "native after_opt final_result.json (every fifth frame, HI-SLAM2 "
                "keyframes, and final frame)"
            ),
        },
        "sequences": {},
    }

    for sequence in TUM_SEQUENCES:
        print(f"\n=== TUM/{sequence} ===", flush=True)
        images = tum_images(args.data_root, sequence)
        jobs = {method: voxel_jobs[(sequence, method)] for method in VOXEL_METHODS}
        metrics = {
            method: common.metric_by_frame(job, images)
            for method, job in jobs.items()
        }
        frame_ids = sorted(set(metrics["ours"]) & set(metrics["ours_mvs"]))
        if not frame_ids:
            raise TumEvaluationError(f"No common voxel keyframes for {sequence}")

        sequence_rows: dict[str, dict[str, Any]] = {
            "hislam2": hislam2_rows[sequence]
        }
        for method in VOXEL_METHODS:
            method_output = output.parent / sequence / method
            if args.skip_lpips:
                lpips = math.nan
            else:
                lpips_by_frame = lpips_for_voxel_frames(
                    jobs[method],
                    metrics[method],
                    frame_ids,
                    args.lpips_device,
                    method_output / "lpips_per_frame.json",
                    runtime,
                )
                lpips = common.mean(lpips_by_frame[frame] for frame in frame_ids)
            sequence_rows[method] = {
                "evaluated_frames": len(frame_ids),
                "frame_ids": frame_ids,
                "psnr": common.mean(metrics[method][frame]["psnr"] for frame in frame_ids),
                "ssim": common.mean(metrics[method][frame]["ssim"] for frame in frame_ids),
                "lpips": lpips,
                "shutdown_dir": str(jobs[method].shutdown_dir),
            }
        summary["sequences"][sequence] = sequence_rows

    summary["average"] = {
        method: average_rows(
            {
                sequence: summary["sequences"][sequence][method]
                for sequence in TUM_SEQUENCES
            }
        )
        for method in ("hislam2", *VOXEL_METHODS)
    }
    common.write_json(output, summary)
    print(f"\nSaved {output}\n", flush=True)
    print(latex_rows(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TumEvaluationError, common.EvaluationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
