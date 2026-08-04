#!/usr/bin/env python3
"""Evaluate rendered-depth, TANDEM MVS, and Omnidata densification runs."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_reconstructions_hislam2 import (  # noqa: E402
    Dataset,
    Experiment,
    HI_SLAM2_EVALUATOR,
    HI_SLAM2_PYTHON,
    MESH_EVAL,
    align_mesh,
    evaluate_aligned_mesh,
    materialize_gt_trajectory,
    require_file,
    result_from_metrics,
    threshold_label,
    write_summaries,
)


DISTANCE_THRESHOLD_M = 0.05
OUTPUT_DIR = (
    REPO_ROOT / "results/monocular_densification_ablation_evaluation"
)


@dataclass(frozen=True)
class Run:
    key: str
    method: str
    dataset_key: str
    dataset: str
    frame_count: int
    directory: Path
    evaluate_reconstruction: bool
    expected_rendered_depth: int
    expected_mvs: int
    expected_omnidata: int


RUNS = (
    Run(
        key="tum_rendered_depth",
        method="Rendered-depth densification",
        dataset_key="tum",
        dataset="TUM fr1/desk",
        frame_count=613,
        directory=REPO_ROOT / (
            "results/tum_voxel/rgbd_dataset_freiburg1_desk/"
            "new_experiments/4641_shutdown"
        ),
        evaluate_reconstruction=False,
        expected_rendered_depth=1,
        expected_mvs=0,
        expected_omnidata=0,
    ),
    Run(
        key="tum_mvs",
        method="TANDEM MVS",
        dataset_key="tum",
        dataset="TUM fr1/desk",
        frame_count=613,
        directory=REPO_ROOT / (
            "results/tum_voxel/rgbd_dataset_freiburg1_desk/"
            "new_experiments/3841_shutdown_MVS"
        ),
        evaluate_reconstruction=False,
        expected_rendered_depth=0,
        expected_mvs=1,
        expected_omnidata=0,
    ),
    Run(
        key="tum_omnidata",
        method="Omnidata",
        dataset_key="tum",
        dataset="TUM fr1/desk",
        frame_count=613,
        directory=REPO_ROOT / (
            "results/tum_voxel/rgbd_dataset_freiburg1_desk/"
            "new_experiments/3841_shutdown_Omni"
        ),
        evaluate_reconstruction=False,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=1,
    ),
    Run(
        key="replica_rendered_depth",
        method="Rendered-depth densification",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_voxel/office0/new_experiments/6810_shutdown"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=1,
        expected_mvs=0,
        expected_omnidata=0,
    ),
    Run(
        key="replica_mvs",
        method="TANDEM MVS",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_voxel/office0/"
            "new_experiments/5801_shutdown_MVS"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=1,
        expected_omnidata=0,
    ),
    Run(
        key="replica_omnidata",
        method="Omnidata",
        dataset_key="replica",
        dataset="Replica office0",
        frame_count=2000,
        directory=REPO_ROOT / (
            "results/replica_voxel/office0/"
            "new_experiments/5848_shutdown_Omni"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=1,
    ),
    Run(
        key="scannet_rendered_depth",
        method="Rendered-depth densification",
        dataset_key="scannet",
        dataset="ScanNet scene0000_00",
        frame_count=5578,
        directory=REPO_ROOT / (
            "results/scannet_voxel/scene0000_00/"
            "new_experiments/18961_shutdown"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=1,
        expected_mvs=0,
        expected_omnidata=0,
    ),
    Run(
        key="scannet_mvs",
        method="TANDEM MVS",
        dataset_key="scannet",
        dataset="ScanNet scene0000_00",
        frame_count=5578,
        directory=REPO_ROOT / (
            "results/scannet_voxel/scene0000_00/"
            "new_experiments/9361_shutdown_MVS"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=1,
        expected_omnidata=0,
    ),
    Run(
        key="scannet_omnidata",
        method="Omnidata",
        dataset_key="scannet",
        dataset="ScanNet scene0000_00",
        frame_count=5578,
        directory=REPO_ROOT / (
            "results/scannet_voxel/scene0000_00/"
            "new_experiments/9361_shutdown_Omni"
        ),
        evaluate_reconstruction=True,
        expected_rendered_depth=0,
        expected_mvs=0,
        expected_omnidata=1,
    ),
)


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def finite_float(value: Any, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {field} in {path}: {value!r}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {field} in {path}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot parse JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def discover_one(root: Path, pattern: str, description: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} below {root}; "
            f"found {len(candidates)}"
        )
    require_file(candidates[0], description)
    return candidates[0].resolve()


def saved_config(run: Run) -> Path:
    return discover_one(run.directory, "*_mono_voxel.yaml", "saved YAML")


def yaml_int(text: str, key: str, default: int = 0) -> int:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*:\s*([01])(?:\s|$)", text
    )
    return default if match is None else int(match.group(1))


def validate_saved_config(run: Run) -> Path:
    path = saved_config(run)
    text = path.read_text(encoding="utf-8")
    actual = (
        yaml_int(text, "Mapper.monocular_rendered_depth_densify"),
        yaml_int(text, "Mapper.monocular_mvs_densify"),
        yaml_int(text, "Mapper.monocular_omnidata_densify"),
    )
    expected = (
        run.expected_rendered_depth,
        run.expected_mvs,
        run.expected_omnidata,
    )
    if actual != expected:
        raise RuntimeError(
            f"Saved YAML does not match {run.method}: {path}; "
            f"expected flags {expected}, found {actual}"
        )
    return path.resolve()


def parse_keyed_metric(path: Path) -> dict[int, float]:
    require_file(path, "per-keyframe metric")
    values: dict[int, float] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise RuntimeError(f"Expected two columns at {path}:{line_number}")
        try:
            keyframe_id = int(tokens[0])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe ID at {path}:{line_number}"
            ) from error
        if keyframe_id in values:
            raise RuntimeError(f"Duplicate keyframe {keyframe_id} in {path}")
        values[keyframe_id] = finite_float(tokens[1], "metric", path)
    if not values:
        raise RuntimeError(f"No metrics found in {path}")
    return values


def keyframe_to_frame(run: Run) -> dict[int, int]:
    path = run.directory / "kf_frame_id_map.txt"
    require_file(path, "keyframe/frame map")
    result: dict[int, int] = {}
    seen_frames: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise RuntimeError(f"Expected two columns at {path}:{line_number}")
        try:
            keyframe_id, frame_id = (int(token) for token in tokens)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe/frame ID at {path}:{line_number}"
            ) from error
        if not 0 <= frame_id < run.frame_count:
            raise RuntimeError(
                f"Frame {frame_id} is outside [0,{run.frame_count}) in {path}"
            )
        if keyframe_id in result or frame_id in seen_frames:
            raise RuntimeError(f"Duplicate keyframe or frame in {path}")
        result[keyframe_id] = frame_id
        seen_frames.add(frame_id)
    if not result:
        raise RuntimeError(f"No mappings found in {path}")
    return result


def metric_by_frame(
    values: dict[int, float], mapping: dict[int, int], path: Path
) -> dict[int, float]:
    missing = sorted(set(values).difference(mapping))
    if missing:
        raise RuntimeError(
            f"{path} contains keyframes absent from kf_frame_id_map.txt: "
            f"{missing[:10]}"
        )
    return {mapping[keyframe_id]: value for keyframe_id, value in values.items()}


def photometric_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    for dataset_key in ("tum", "replica", "scannet"):
        runs = [run for run in RUNS if run.dataset_key == dataset_key]
        values_by_run: dict[str, dict[int, dict[str, float]]] = {}
        for run in runs:
            mapping = keyframe_to_frame(run)
            paths = {
                "psnr": run.directory / "psnr.txt",
                "ssim": run.directory / "dssim.txt",
                "render_ms": run.directory / "render_time.txt",
            }
            metrics = {
                name: metric_by_frame(parse_keyed_metric(path), mapping, path)
                for name, path in paths.items()
            }
            frame_ids = set.intersection(*(set(value) for value in metrics.values()))
            if any(set(value) != frame_ids for value in metrics.values()):
                raise RuntimeError(f"Metric frame sets differ for {run.key}")
            values_by_run[run.key] = {
                frame_id: {
                    name: metric[frame_id] for name, metric in metrics.items()
                }
                for frame_id in frame_ids
            }

        common_frames = set.intersection(
            *(set(values_by_run[run.key]) for run in runs)
        )
        if not common_frames:
            raise RuntimeError(f"No exact common frames for {runs[0].dataset}")
        for run in runs:
            rows = [
                values_by_run[run.key][frame_id]
                for frame_id in sorted(common_frames)
            ]
            summaries.append(
                {
                    "dataset": run.dataset,
                    "method": run.method,
                    "evaluated_frames": len(rows),
                    "psnr": statistics.fmean(row["psnr"] for row in rows),
                    "ssim": statistics.fmean(row["ssim"] for row in rows),
                    "render_fps_hz": 1000.0
                    / statistics.fmean(row["render_ms"] for row in rows),
                    "run_dir": str(run.directory),
                }
            )
            for frame_id in sorted(common_frames):
                per_frame.append(
                    {
                        "dataset": run.dataset,
                        "method": run.method,
                        "dataset_frame_id": frame_id,
                        **values_by_run[run.key][frame_id],
                    }
                )
    return summaries, per_frame


def native_map(run: Run) -> Path:
    return discover_one(
        run.directory,
        "ply/voxel_model/iteration_*/voxel_model.ply",
        "native voxel map",
    )


def surface_mesh(run: Run) -> Path:
    return discover_one(
        run.directory,
        "ply/voxel_model/iteration_*/voxel_surface_mesh.ply",
        "voxel surface mesh",
    )


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as ply_file:
        for raw_line in ply_file:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise RuntimeError(f"PLY has no element vertex declaration: {path}")


def shutdown_iteration(run: Run) -> int:
    match = re.match(r"^(\d+)_shutdown", run.directory.name)
    if match is None:
        raise RuntimeError(f"Cannot parse shutdown iteration: {run.directory}")
    return int(match.group(1))


def computational_result(run: Run, config: Path) -> dict[str, Any]:
    map_path = native_map(run)
    runtime_path = run.directory / "runtime_metrics.json"
    runtime: dict[str, Any] | None = None
    if runtime_path.is_file():
        runtime = load_json(runtime_path)
        frames = int(runtime.get("frames", 0))
        if frames != run.frame_count:
            raise RuntimeError(
                f"Expected {run.frame_count} frames in {runtime_path}; "
                f"found {frames}"
            )
    return {
        "dataset": run.dataset,
        "method": run.method,
        "frames": run.frame_count,
        "keyframes": len(keyframe_to_frame(run)),
        "iterations": (
            int(runtime["iterations"]) if runtime is not None
            else shutdown_iteration(run)
        ),
        "voxels": ply_vertex_count(map_path),
        "runtime_seconds": (
            finite_float(runtime["total_seconds"], "total_seconds", runtime_path)
            if runtime is not None else None
        ),
        "system_fps_hz": (
            run.frame_count
            / finite_float(runtime["total_seconds"], "total_seconds", runtime_path)
            if runtime is not None else None
        ),
        "map_size_mb": map_path.stat().st_size / (1024.0 * 1024.0),
        "gpu_memory_allocated_mb": (
            finite_float(
                runtime["gpu_memory_allocated_mb"],
                "gpu_memory_allocated_mb",
                runtime_path,
            )
            if runtime is not None else None
        ),
        "gpu_memory_reserved_mb": (
            finite_float(
                runtime["gpu_memory_reserved_mb"],
                "gpu_memory_reserved_mb",
                runtime_path,
            )
            if runtime is not None else None
        ),
        "runtime_available": runtime is not None,
        "saved_config": str(config),
        "native_map": str(map_path),
    }


def reconstruction_experiment(run: Run, dataset_tag: str) -> Experiment:
    trajectory = run.directory / "CameraTrajectory_TUM.txt"
    require_file(trajectory, "run-local reconstruction trajectory")
    return Experiment(
        name=run.method,
        family="SVRecon monocular",
        directory=(
            OUTPUT_DIR / "reconstruction" / run.dataset_key
            / dataset_tag / run.key
        ),
        mesh=surface_mesh(run),
        recon_trajectory=trajectory.resolve(),
    )


def reconstruction_datasets() -> tuple[Dataset, ...]:
    replica_runs = [run for run in RUNS if run.dataset_key == "replica"]
    scannet_runs = [run for run in RUNS if run.dataset_key == "scannet"]
    replica_gt_trajectory = REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
    replica_placeholder = replica_runs[0].directory / "CameraTrajectory_TUM.txt"
    scannet_placeholder = scannet_runs[0].directory / "CameraTrajectory_TUM.txt"
    return (
        Dataset(
            name="Replica office0 (culled GT)",
            gt_mesh=(
                REPO_ROOT
                / "third_party/ESLAM/cull_replica_mesh/office0_culled.ply"
            ),
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=replica_placeholder,
            artifact_tag="culled",
            summary_dir=OUTPUT_DIR / "reconstruction/replica_culled",
            experiments=tuple(
                reconstruction_experiment(run, "culled")
                for run in replica_runs
            ),
        ),
        Dataset(
            name="Replica office0 (original GT)",
            gt_mesh=REPO_ROOT / "scripts/data/Replica/office0_mesh.ply",
            gt_trajectory=replica_gt_trajectory,
            shared_recon_trajectory=replica_placeholder,
            artifact_tag="original",
            summary_dir=OUTPUT_DIR / "reconstruction/replica_original",
            experiments=tuple(
                reconstruction_experiment(run, "original")
                for run in replica_runs
            ),
        ),
        Dataset(
            name="ScanNet scene0000_00",
            gt_mesh=(
                REPO_ROOT / "scripts/data/ScanNet/scans/scene0000_00"
                / "scene0000_00_vh_clean_2.ply"
            ),
            gt_pose_directory=(
                REPO_ROOT / "scripts/data/ScanNet/scans/scene0000_00/pose"
            ),
            shared_recon_trajectory=scannet_placeholder,
            summary_dir=OUTPUT_DIR / "reconstruction/scannet",
            experiments=tuple(
                reconstruction_experiment(run, "scene0000_00")
                for run in scannet_runs
            ),
        ),
    )


def evaluate_reconstruction() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset in reconstruction_datasets():
        print(f"\n=== Reconstruction: {dataset.name} ===")
        require_file(dataset.gt_mesh, f"GT mesh for {dataset.name}")
        gt_trajectory = materialize_gt_trajectory(dataset)
        dataset_results: list[dict[str, Any]] = []
        for experiment in dataset.experiments:
            assert experiment.mesh is not None
            assert experiment.recon_trajectory is not None
            experiment.directory.mkdir(parents=True, exist_ok=True)
            aligned_mesh = align_mesh(
                dataset,
                experiment,
                experiment.mesh,
                experiment.recon_trajectory,
                gt_trajectory,
                DISTANCE_THRESHOLD_M,
            )
            metrics, result_path = evaluate_aligned_mesh(
                dataset,
                experiment,
                aligned_mesh,
                DISTANCE_THRESHOLD_M,
            )
            result = result_from_metrics(
                dataset,
                experiment,
                experiment.mesh,
                aligned_mesh,
                experiment.recon_trajectory,
                result_path,
                metrics,
                DISTANCE_THRESHOLD_M,
            )
            dataset_results.append(result)
            results.append(result)
        write_summaries(
            dataset,
            gt_trajectory,
            dataset_results,
            DISTANCE_THRESHOLD_M,
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, protocol: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"protocol": protocol, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def optional_number(value: Any, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def photometric_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Method | Common frames | PSNR ↑ | SSIM ↑ | Render FPS ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['evaluated_frames']} | {row['psnr']:.4f} | "
            f"{row['ssim']:.4f} | {row['render_fps_hz']:.2f} |"
        )
    return "\n".join(lines)


def computational_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Method | Iterations | Voxels | System FPS ↑ | "
        "Map MB ↓ | GPU reserved MB ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['iterations']} | "
            f"{row['voxels']} | {optional_number(row['system_fps_hz'])} | "
            f"{row['map_size_mb']:.2f} | "
            f"{optional_number(row['gpu_memory_reserved_mb'])} |"
        )
    return "\n".join(lines)


def reconstruction_table(rows: list[dict[str, Any]]) -> str:
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    lines = [
        "| Dataset | Method | Acc. m ↓ | Compl. m ↓ | Chamfer m ↓ | "
        f"Prec@{suffix} ↑ | Recall@{suffix} ↑ | Completion@{suffix} ↑ | "
        f"F@{suffix} ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['name']} | "
            f"{row['accuracy_m']:.6f} | {row['completeness_m']:.6f} | "
            f"{row['chamfer_l1_m']:.6f} | "
            f"{row[f'precision_{suffix}']:.4f} | "
            f"{row[f'recall_{suffix}']:.4f} | "
            f"{row[f'completion_{suffix}_percent']:.2f}% | "
            f"{row[f'fscore_{suffix}']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    require_file(MESH_EVAL, "mesh_eval executable")
    require_file(HI_SLAM2_EVALUATOR, "HI-SLAM2 evaluator")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python")
    configs: dict[str, Path] = {}
    for run in RUNS:
        require_directory(run.directory, "ablation run directory")
        configs[run.key] = validate_saved_config(run)

    photometric, photometric_per_frame = photometric_results()
    computational = [
        computational_result(run, configs[run.key]) for run in RUNS
    ]
    reconstruction = evaluate_reconstruction()

    photo_md = photometric_table(photometric)
    compute_md = computational_table(computational)
    reconstruction_md = reconstruction_table(reconstruction)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT_DIR / "photometric_metrics.csv", photometric)
    write_csv(
        OUTPUT_DIR / "photometric_metrics_per_frame.csv",
        photometric_per_frame,
    )
    write_json(
        OUTPUT_DIR / "photometric_metrics.json",
        {
            "aggregation": "arithmetic mean over exact common dataset frames",
            "ssim_source": "dssim.txt stores SSIM despite its legacy filename",
        },
        photometric,
    )
    (OUTPUT_DIR / "photometric_metrics.md").write_text(
        photo_md + "\n", encoding="utf-8"
    )

    write_csv(OUTPUT_DIR / "computational_metrics.csv", computational)
    write_json(
        OUTPUT_DIR / "computational_metrics.json",
        {
            "system_fps": "dataset frames / recorded mapping total_seconds",
            "map_size": "actual run-local voxel_model.ply size",
            "gpu_memory": "recorded PyTorch CUDA allocator value",
            "missing_runtime": (
                "TUM rendered-depth run has no runtime_metrics.json; "
                "runtime and GPU values are null"
            ),
        },
        computational,
    )
    (OUTPUT_DIR / "computational_metrics.md").write_text(
        compute_md + "\n", encoding="utf-8"
    )

    write_csv(OUTPUT_DIR / "reconstruction_metrics.csv", reconstruction)
    write_json(
        OUTPUT_DIR / "reconstruction_metrics.json",
        {
            "alignment": "trajectory Sim(3), followed by HI-SLAM2 rigid ICP",
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "completion_rate": "identical to recall and reported as percent",
            "tum": "omitted because no GT reconstruction mesh is provided",
        },
        reconstruction,
    )
    (OUTPUT_DIR / "reconstruction_metrics.md").write_text(
        reconstruction_md + "\n", encoding="utf-8"
    )

    summary = (
        "# Monocular Densification Ablation\n\n"
        "## Photometric Quality\n\n"
        f"{photo_md}\n\n"
        "## Computational Performance\n\n"
        f"{compute_md}\n\n"
        "## Reconstruction\n\n"
        f"{reconstruction_md}\n"
    )
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\nSaved ablation evaluation to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
