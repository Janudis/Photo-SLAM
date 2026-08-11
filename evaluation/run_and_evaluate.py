#!/usr/bin/env python3
"""Run one existing Photo-SLAM shell runner and evaluate its new shutdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_monocular_densification_ablation import (  # noqa: E402
    Run as AblationRun,
    evaluate_replica_primitive_floaters,
    evaluate_replica_surface_extras,
    save_hi_icp_transform,
)
from evaluation.evaluate_reconstructions_hislam2 import (  # noqa: E402
    Dataset,
    Experiment,
    HI_SLAM2_EVALUATOR,
    HI_SLAM2_PYTHON,
    MESH_EVAL,
    align_mesh,
    discover_mesh,
    evaluate_aligned_mesh,
    materialize_gt_trajectory,
    require_file,
    result_from_metrics,
    threshold_label,
    write_summaries,
)


RESULTS_ROOT = REPO_ROOT / "results"
DISTANCE_THRESHOLD_M = 0.05
SHUTDOWN_PATTERN = re.compile(r"^\d+_shutdown$")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    frame_count: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an existing dataset .sh file unchanged, then evaluate the "
            "shutdown it creates."
        )
    )
    parser.add_argument("runner", type=Path, help="existing dataset .sh file")
    return parser.parse_args()


def resolve_runner(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    require_file(candidate, "dataset runner")
    if candidate.suffix != ".sh":
        raise ValueError(f"Runner must be a .sh file: {candidate}")
    return candidate


def file_signature(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def shutdown_signature(path: Path) -> tuple[Any, ...]:
    stat = path.stat()
    tracked_files = (
        "runtime_metrics.json",
        "psnr.txt",
        "dssim.txt",
        "CameraTrajectory_TUM.txt",
        "kf_frame_id_map.txt",
    )
    return (
        stat.st_mtime_ns,
        *(file_signature(path / name) for name in tracked_files),
    )


def snapshot_shutdowns() -> dict[Path, tuple[Any, ...]]:
    if not RESULTS_ROOT.is_dir():
        return {}
    result: dict[Path, tuple[Any, ...]] = {}
    for path in RESULTS_ROOT.rglob("*_shutdown"):
        if not path.is_dir() or SHUTDOWN_PATTERN.fullmatch(path.name) is None:
            continue
        resolved = path.resolve()
        result[resolved] = shutdown_signature(resolved)
    return result


def changed_shutdown(
    before: dict[Path, tuple[Any, ...]],
) -> Path:
    after = snapshot_shutdowns()
    changed = sorted(
        path for path, signature in after.items()
        if before.get(path) != signature
    )
    if len(changed) != 1:
        rendered = ", ".join(str(path) for path in changed)
        raise RuntimeError(
            "Expected the runner to create or update exactly one numeric "
            f"*_shutdown directory; found {len(changed)}"
            + (f": {rendered}" if rendered else "")
        )
    return changed[0]


def run_streamed(runner: Path, log_path: Path) -> None:
    command = [str(runner)]
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[cwd={REPO_ROOT}] $ {rendered}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Runner failed with exit code {return_code}; see {log_path}"
        )


def infer_dataset(shutdown_dir: Path) -> DatasetSpec:
    try:
        result_family = shutdown_dir.relative_to(RESULTS_ROOT).parts[0].lower()
    except (ValueError, IndexError) as error:
        raise RuntimeError(
            f"Shutdown is not below the repository results folder: {shutdown_dir}"
        ) from error

    if result_family.startswith("replica"):
        return DatasetSpec("replica", "Replica office0", 2000)
    if result_family.startswith("scannet"):
        return DatasetSpec("scannet", "ScanNet scene0000_00", 5578)
    if result_family.startswith("tum"):
        return DatasetSpec("tum", "TUM fr1/desk", 613)
    if result_family.startswith("euroc"):
        sequence = shutdown_dir.parent.name
        return DatasetSpec("euroc", f"EuRoC {sequence}")
    if result_family.startswith("waymo"):
        return DatasetSpec("waymo", "Waymo")
    raise RuntimeError(
        f"Cannot infer the dataset from result path: {shutdown_dir}"
    )


def finite_float(value: str, path: Path, line_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise RuntimeError(
            f"Invalid metric at {path}:{line_number}: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite metric at {path}:{line_number}")
    return result


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
            raise RuntimeError(
                f"Expected two columns at {path}:{line_number}"
            )
        try:
            keyframe_id = int(tokens[0])
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe ID at {path}:{line_number}"
            ) from error
        if keyframe_id in values:
            raise RuntimeError(
                f"Duplicate keyframe {keyframe_id} in {path}"
            )
        values[keyframe_id] = finite_float(tokens[1], path, line_number)
    if not values:
        raise RuntimeError(f"No metrics found in {path}")
    return values


def load_keyframe_frame_map(
    shutdown_dir: Path, keyframes: set[int]
) -> dict[int, int]:
    path = shutdown_dir / "kf_frame_id_map.txt"
    require_file(path, "keyframe/frame map")
    result: dict[int, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2:
            raise RuntimeError(
                f"Expected two columns at {path}:{line_number}"
            )
        try:
            keyframe_id, frame_id = (int(token) for token in tokens)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid keyframe/frame ID at {path}:{line_number}"
            ) from error
        if keyframe_id in result:
            raise RuntimeError(f"Duplicate keyframe {keyframe_id} in {path}")
        result[keyframe_id] = frame_id
    missing = sorted(keyframes.difference(result))
    if missing:
        raise RuntimeError(
            f"Metrics reference keyframes absent from {path}: {missing[:10]}"
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_photometric(
    dataset: DatasetSpec,
    shutdown_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    psnr = parse_keyed_metric(shutdown_dir / "psnr.txt")
    ssim = parse_keyed_metric(shutdown_dir / "dssim.txt")
    if set(psnr) != set(ssim):
        raise RuntimeError(
            f"PSNR and SSIM keyframe sets differ in {shutdown_dir}"
        )
    frame_map = load_keyframe_frame_map(shutdown_dir, set(psnr))
    per_frame = [
        {
            "keyframe_id": keyframe_id,
            "dataset_frame_id": frame_map[keyframe_id],
            "psnr": psnr[keyframe_id],
            "ssim": ssim[keyframe_id],
        }
        for keyframe_id in sorted(psnr)
    ]
    summary = {
        "dataset": dataset.name,
        "experiment": shutdown_dir.name,
        "evaluated_keyframes": len(per_frame),
        "psnr": statistics.fmean(row["psnr"] for row in per_frame),
        "ssim": statistics.fmean(row["ssim"] for row in per_frame),
        "aggregation": "arithmetic mean over this run's rendered keyframes",
        "shutdown_dir": str(shutdown_dir),
    }
    write_csv(output_dir / "photometric_per_frame.csv", per_frame)
    write_csv(output_dir / "photometric_summary.csv", [summary])
    write_json(output_dir / "photometric_summary.json", summary)
    markdown = (
        "| Dataset | Experiment | Keyframes | PSNR ↑ | SSIM ↑ |\n"
        "|---|---|---:|---:|---:|\n"
        f"| {dataset.name} | {shutdown_dir.name} | {len(per_frame)} | "
        f"{summary['psnr']:.4f} | {summary['ssim']:.4f} |\n"
    )
    (output_dir / "photometric_summary.md").write_text(
        markdown, encoding="utf-8"
    )
    return summary


def reconstruction_datasets(
    dataset: DatasetSpec,
    mesh: Path,
    trajectory: Path,
    output_dir: Path,
) -> tuple[Dataset, ...]:
    experiment_name = trajectory.parent.name
    if dataset.key == "replica":
        gt_trajectory = REPO_ROOT / "scripts/data/Replica/office0/traj.txt"
        definitions = (
            (
                "Replica office0 (culled GT)",
                REPO_ROOT
                / "third_party/ESLAM/cull_replica_mesh/office0_culled.ply",
                "culled",
            ),
            (
                "Replica office0 (original GT)",
                REPO_ROOT / "scripts/data/Replica/office0_mesh.ply",
                "original",
            ),
        )
        result: list[Dataset] = []
        for name, gt_mesh, tag in definitions:
            experiment = Experiment(
                name=experiment_name,
                family="SVRecon",
                directory=output_dir / "reconstruction" / tag / "run",
                mesh=mesh,
                recon_trajectory=trajectory,
            )
            result.append(
                Dataset(
                    name=name,
                    gt_mesh=gt_mesh,
                    gt_trajectory=gt_trajectory,
                    shared_recon_trajectory=trajectory,
                    artifact_tag=tag,
                    summary_dir=(
                        output_dir / "reconstruction" / tag / "summary"
                    ),
                    experiments=(experiment,),
                )
            )
        return tuple(result)

    if dataset.key == "scannet":
        data_root = REPO_ROOT / "scripts/data/ScanNet/scans/scene0000_00"
        experiment = Experiment(
            name=experiment_name,
            family="SVRecon",
            directory=output_dir / "reconstruction/scannet/run",
            mesh=mesh,
            recon_trajectory=trajectory,
        )
        return (
            Dataset(
                name="ScanNet scene0000_00",
                gt_mesh=data_root / "scene0000_00_vh_clean_2.ply",
                gt_pose_directory=data_root / "pose",
                shared_recon_trajectory=trajectory,
                summary_dir=output_dir / "reconstruction/scannet/summary",
                experiments=(experiment,),
            ),
        )
    return ()


def evaluate_reconstruction(
    dataset: DatasetSpec,
    shutdown_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if dataset.key not in {"replica", "scannet"}:
        return []

    require_file(MESH_EVAL, "mesh_eval executable")
    require_file(HI_SLAM2_EVALUATOR, "HI-SLAM2 reconstruction evaluator")
    require_file(HI_SLAM2_PYTHON, "HI-SLAM2 Python interpreter")
    trajectory = shutdown_dir / "CameraTrajectory_TUM.txt"
    require_file(trajectory, "run-local reconstruction trajectory")
    mesh = discover_mesh(
        Experiment(
            name=shutdown_dir.name,
            family="SVRecon",
            directory=shutdown_dir,
        )
    )
    datasets = reconstruction_datasets(
        dataset, mesh, trajectory.resolve(), output_dir
    )
    results: list[dict[str, Any]] = []
    replica_run: AblationRun | None = None
    if dataset.key == "replica":
        assert dataset.frame_count is not None
        replica_run = AblationRun(
            key=shutdown_dir.name,
            method=shutdown_dir.name,
            dataset_key="replica",
            dataset=dataset.name,
            frame_count=dataset.frame_count,
            directory=shutdown_dir,
            evaluate_reconstruction=True,
            expected_rendered_depth=0,
            expected_mvs=0,
            expected_omnidata=0,
        )

    for reconstruction_dataset in datasets:
        require_file(
            reconstruction_dataset.gt_mesh,
            f"GT mesh for {reconstruction_dataset.name}",
        )
        gt_trajectory = materialize_gt_trajectory(reconstruction_dataset)
        experiment = reconstruction_dataset.experiments[0]
        assert experiment.mesh is not None
        assert experiment.recon_trajectory is not None
        experiment.directory.mkdir(parents=True, exist_ok=True)
        aligned_mesh = align_mesh(
            reconstruction_dataset,
            experiment,
            experiment.mesh,
            experiment.recon_trajectory,
            gt_trajectory,
            DISTANCE_THRESHOLD_M,
        )
        metrics, result_path = evaluate_aligned_mesh(
            reconstruction_dataset,
            experiment,
            aligned_mesh,
            DISTANCE_THRESHOLD_M,
        )
        standard_result = result_from_metrics(
            reconstruction_dataset,
            experiment,
            experiment.mesh,
            aligned_mesh,
            experiment.recon_trajectory,
            result_path,
            metrics,
            DISTANCE_THRESHOLD_M,
        )
        write_summaries(
            reconstruction_dataset,
            gt_trajectory,
            [standard_result],
            DISTANCE_THRESHOLD_M,
        )
        result = dict(standard_result)

        if (
            dataset.key == "replica"
            and reconstruction_dataset.artifact_tag == "culled"
        ):
            assert replica_run is not None
            hi_icp_transform = save_hi_icp_transform(
                result_path.with_suffix(".log"),
                experiment.directory / "hi_icp_transform.txt",
            )
            surface_extras = evaluate_replica_surface_extras(
                reconstruction_dataset,
                experiment,
                aligned_mesh,
                gt_trajectory,
            )
            primitive_extras = evaluate_replica_primitive_floaters(
                replica_run,
                reconstruction_dataset,
                experiment,
                experiment.recon_trajectory,
                gt_trajectory,
                hi_icp_transform,
            )
            result.update(
                {
                    "depth_l1_m": surface_extras["depth_l1_m"],
                    "depth_l1_cm": 100.0 * surface_extras["depth_l1_m"],
                    "mesh_floater_ratio": surface_extras[
                        "mesh_floater_ratio"
                    ],
                    "mesh_floater_percent": (
                        100.0 * surface_extras["mesh_floater_ratio"]
                    ),
                    "primitive_floater_ratio": primitive_extras[
                        "primitive_floater_ratio"
                    ],
                    "primitive_floater_percent": (
                        100.0
                        * primitive_extras["primitive_floater_ratio"]
                    ),
                }
            )
        results.append(result)

    write_csv(output_dir / "reconstruction_summary.csv", results)
    write_json(
        output_dir / "reconstruction_summary.json",
        {
            "protocol": {
                "alignment": "trajectory Sim(3), then HI-SLAM2 rigid ICP",
                "distance_threshold_m": DISTANCE_THRESHOLD_M,
            },
            "results": results,
        },
    )
    suffix = threshold_label(DISTANCE_THRESHOLD_M)
    lines = [
        "| Dataset | Acc. m ↓ | Compl. m ↓ | "
        f"Prec@{suffix} ↑ | Recall@{suffix} ↑ | "
        f"Completion@{suffix} ↑ | F@{suffix} ↑ | "
        "Depth L1 cm ↓ | Mesh floaters ↓ | Primitive floaters ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        depth = result.get("depth_l1_cm")
        mesh_floaters = result.get("mesh_floater_percent")
        primitive_floaters = result.get("primitive_floater_percent")
        extras = "N/A | N/A | N/A"
        if (
            depth is not None
            and mesh_floaters is not None
            and primitive_floaters is not None
        ):
            extras = (
                f"{depth:.4f} | {mesh_floaters:.2f}% | "
                f"{primitive_floaters:.2f}%"
            )
        lines.append(
            f"| {result['dataset']} | {result['accuracy_m']:.6f} | "
            f"{result['completeness_m']:.6f} | "
            f"{result[f'precision_{suffix}']:.4f} | "
            f"{result[f'recall_{suffix}']:.4f} | "
            f"{result[f'completion_{suffix}_percent']:.2f}% | "
            f"{result[f'fscore_{suffix}']:.4f} | {extras} |"
        )
    (output_dir / "reconstruction_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return results


def saved_configs(shutdown_dir: Path) -> list[str]:
    return [str(path.resolve()) for path in sorted(shutdown_dir.glob("*.yaml"))]


def main() -> int:
    args = parse_args()
    runner = resolve_runner(args.runner)
    before = snapshot_shutdowns()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    temporary_log = Path(tempfile.gettempdir()) / (
        f"photoslam_run_and_evaluate_{timestamp}.log"
    )

    print(f"\n=== Running {runner.name} ===")
    run_streamed(runner, temporary_log)
    shutdown_dir = changed_shutdown(before)
    dataset = infer_dataset(shutdown_dir)
    output_dir = shutdown_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temporary_log, output_dir / "mapping.log")
    write_json(
        output_dir / "run_metadata.json",
        {
            "dataset": dataset.name,
            "runner": str(runner),
            "saved_configs": saved_configs(shutdown_dir),
            "shutdown_dir": str(shutdown_dir),
        },
    )

    print("\n=== Photometric evaluation ===")
    photometric = evaluate_photometric(
        dataset, shutdown_dir, output_dir
    )
    print(
        f"PSNR={photometric['psnr']:.4f}, "
        f"SSIM={photometric['ssim']:.4f}"
    )

    print("\n=== Reconstruction evaluation ===")
    reconstruction = evaluate_reconstruction(
        dataset, shutdown_dir, output_dir
    )
    if not reconstruction:
        print(f"{dataset.name}: no compatible GT mesh; skipped")

    summary_lines = [
        f"# {shutdown_dir.name}",
        "",
        f"Dataset: {dataset.name}",
        f"Runner: `{runner}`",
        f"Shutdown: `{shutdown_dir}`",
        "",
        "## Photometric",
        "",
        (output_dir / "photometric_summary.md").read_text(
            encoding="utf-8"
        ).strip(),
    ]
    if reconstruction:
        summary_lines.extend(
            [
                "",
                "## Reconstruction",
                "",
                (output_dir / "reconstruction_summary.md").read_text(
                    encoding="utf-8"
                ).strip(),
            ]
        )
    else:
        summary_lines.extend(
            ["", "## Reconstruction", "", "No compatible GT mesh."]
        )
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(f"\nCompleted run: {shutdown_dir}")
    print(f"Evaluation:    {output_dir}")
    print(f"Summary:       {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
