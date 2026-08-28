#!/usr/bin/env python3
"""Run and evaluate Replica office0 MVS-TSDF pruning ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import evaluate as benchmark_evaluate
import run as benchmark_run


REPO_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "office0"

# Keep MVS supervision fixed while isolating pruning behavior. The full
# Replica presets use this same baseline after the office0 selection study.
COMMON_OVERRIDES = dict(benchmark_run.MVS_TSDF_COMMON_OVERRIDES)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    family: str
    description: str
    changes: dict[str, int | float]

    @property
    def overrides(self) -> dict[str, int | float]:
        return {**COMMON_OVERRIDES, **self.changes}

    @property
    def requires_mvs(self) -> bool:
        return True


VARIANTS: dict[str, Variant] = {
    "sdf_only": Variant(
        key="sdf_only",
        label="Scheduled SDF pruning only",
        family="pruning",
        description=(
            "Uses scheduled SVRecon SDF, near-camera, and far-range pruning "
            "without co-visibility, MVS consistency, or final refinement."
        ),
        changes={},
    ),
    "surface_views": Variant(
        key="surface_views",
        label="Co-visibility pruning",
        family="pruning",
        description=(
            "Adds MonoGS-style occlusion-aware multi-view support to the "
            "scheduled SDF, near-camera, and far-range rules."
        ),
        changes={"Optimization.prune_surface_views_enable": 1},
    ),
    "surface_views_final": Variant(
        key="surface_views_final",
        label="Table pruning switches",
        family="pruning",
        description=(
            "Uses the pruning switches from the table runs: co-visibility "
            "and final refinement enabled, MVS consistency off. The fixed "
            "MVS-TSDF densification and depth loss still differ from those runs."
        ),
        changes={
            "Optimization.prune_surface_views_enable": 1,
            "Optimization.final_refinement_enable": 1,
        },
    ),
    "mvs_consistency": Variant(
        key="mvs_consistency",
        label="MVS-consistency pruning",
        family="pruning",
        description=(
            "Uses MVS depth to protect supported SDF candidates and add "
            "multi-view free-space pruning candidates, without co-visibility "
            "or final refinement."
        ),
        changes={"Optimization.prune_mvs_consistency_enable": 1},
    ),
    "combined": Variant(
        key="combined",
        label="Co-visibility + MVS consistency",
        family="pruning",
        description=(
            "Combines renderer co-visibility with MVS support protection and "
            "multi-view free-space evidence, without final refinement."
        ),
        changes={
            "Optimization.prune_surface_views_enable": 1,
            "Optimization.prune_mvs_consistency_enable": 1,
        },
    ),
    "combined_final": Variant(
        key="combined_final",
        label="Combined + final refinement",
        family="pruning",
        description=(
            "Runs final refinement after the combined online co-visibility "
            "and MVS-consistency configuration."
        ),
        changes={
            "Optimization.prune_surface_views_enable": 1,
            "Optimization.prune_mvs_consistency_enable": 1,
            "Optimization.final_refinement_enable": 1,
        },
    ),
}

SUITES: dict[str, tuple[str, ...]] = {
    "core": (
        "sdf_only",
        "surface_views",
        "surface_views_final",
        "mvs_consistency",
        "combined",
        "combined_final",
    ),
}

METRIC_DIRECTIONS = {
    "accuracy_cm": "min",
    "completeness_cm": "min",
    "completion_ratio_percent": "max",
    "precision_5cm": "max",
    "recall_5cm": "max",
    "fscore_5cm": "max",
    "psnr": "max",
    "ssim": "max",
    "lpips": "min",
}


class AblationError(RuntimeError):
    pass


def default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"replica-office0-mvs-tsdf-pruning-{timestamp}"


def selected_variants(args: argparse.Namespace) -> tuple[Variant, ...]:
    keys = args.variants or list(SUITES[args.suite])
    return tuple(VARIANTS[key] for key in dict.fromkeys(keys))


def suite_name(args: argparse.Namespace) -> str:
    return "custom" if args.variants else args.suite


def validate_variant_matrix(variants: Iterable[Variant]) -> None:
    for variant in variants:
        values = variant.overrides
        modes = (
            int(bool(values["Mapper.monocular_rendered_depth_densify"])),
            int(bool(values["Mapper.monocular_mvs_densify"])),
            int(bool(values["Mapper.monocular_mvs_tsdf_evidence"])),
        )
        if modes != (0, 0, 1):
            raise AblationError(
                f"Variant {variant.key} does not exclusively use MVS TSDF"
            )
        if values["Mapper.monocular_mvs_tsdf_evidence_trunc_vox"] != 2.0:
            raise AblationError(f"Variant {variant.key} does not use trunc2")
        if values["Optimization.lambda_monocular_depth"] <= 0.0:
            raise AblationError(
                f"Variant {variant.key} disables depth regularization"
            )


def variant_manifest(variant: Variant) -> dict[str, Any]:
    return {
        "key": variant.key,
        "label": variant.label,
        "family": variant.family,
        "description": variant.description,
        "overrides": variant.overrides,
    }


def require_evaluation_environment(args: argparse.Namespace) -> None:
    benchmark_run.require_file(
        REPO_ROOT / "bin/mesh_eval", "mesh_eval executable"
    )
    benchmark_run.require_file(
        REPO_ROOT / "third_party/HI-SLAM2/scripts/eval_recon.py",
        "HI-SLAM2 reconstruction evaluator",
    )
    geometry_check = subprocess.run(
        [
            args.hislam_python,
            "-c",
            "import open3d, trimesh, evaluate_3d_reconstruction",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if geometry_check.returncode != 0:
        raise AblationError(
            "HI-SLAM2 evaluation dependencies are unavailable:\n"
            + geometry_check.stdout.strip()
        )
    if not args.skip_lpips:
        lpips_check = subprocess.run(
            [sys.executable, "-c", "import lpips; from PIL import Image"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if lpips_check.returncode != 0:
            raise AblationError(
                "LPIPS evaluation dependencies are unavailable:\n"
                + lpips_check.stdout.strip()
            )


def make_jobs(
    variants: Iterable[Variant],
    repetitions: int,
    data_dir: Path,
    run_root: Path,
) -> list[benchmark_run.Job]:
    dataset = benchmark_run.DATASETS["replica"]
    mapper_config = REPO_ROOT / dataset.voxel_config
    orb_config = REPO_ROOT / benchmark_run.relative_config_path(
        dataset.orb_config, SEQUENCE
    )
    binary = REPO_ROOT / dataset.voxel_binary
    benchmark_run.require_file(mapper_config, "Replica voxel configuration")
    benchmark_run.require_file(orb_config, "Replica ORB-SLAM configuration")
    benchmark_run.require_file(binary, "Replica voxel executable")

    source_text = mapper_config.read_text(encoding="utf-8")
    jobs: list[benchmark_run.Job] = []
    for variant in variants:
        # Validate that every controlled key exists exactly once before any run.
        benchmark_run.apply_yaml_overrides(source_text, variant.overrides)
        method = benchmark_run.MethodSpec(
            key=variant.key,
            label=variant.label,
            family="voxel",
            voxel_overrides=variant.overrides,
            requires_mvs_model=variant.requires_mvs,
        )
        for trial in range(1, repetitions + 1):
            jobs.append(
                benchmark_run.Job(
                    dataset=dataset,
                    sequence=SEQUENCE,
                    method=method,
                    trial=trial,
                    data_dir=data_dir,
                    output_dir=(
                        run_root
                        / "replica"
                        / SEQUENCE
                        / variant.key
                        / f"trial_{trial:02d}"
                    ),
                    orb_config=orb_config,
                    mapper_config=mapper_config,
                    photoslam_config=None,
                    binary=binary,
                )
            )
    return jobs


def run_ablation(
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
    jobs: list[benchmark_run.Job],
    run_root: Path,
) -> dict[str, Any]:
    repository = benchmark_run.repository_provenance()
    machine = benchmark_run.machine_provenance()
    manifest_path = run_root / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "dataset": "replica",
        "sequence": SEQUENCE,
        "suite": suite_name(args),
        "repetitions": args.repetitions,
        "variants": [variant_manifest(variant) for variant in variants],
        "repository": repository,
        "machine": machine,
        "jobs": [],
    }
    benchmark_run.write_json(manifest_path, manifest)
    failures = 0
    for job in jobs:
        try:
            result = benchmark_run.run_job(
                job, args.run_id, repository, machine
            )
            manifest["jobs"].append(
                {
                    "status": "complete",
                    "variant": job.method.key,
                    "trial": job.trial,
                    "provenance": str(job.output_dir / "provenance.json"),
                    "shutdown_dir": result["shutdown_dir"],
                }
            )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = benchmark_run.utc_string()
            benchmark_run.write_json(manifest_path, manifest)
            raise
        except Exception as error:
            failures += 1
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            manifest["jobs"].append(
                {
                    "status": "failed",
                    "variant": job.method.key,
                    "trial": job.trial,
                    "provenance": str(job.output_dir / "provenance.json"),
                    "error": str(error),
                }
            )
            if not args.continue_on_error:
                manifest["status"] = "failed"
                manifest["finished_at"] = benchmark_run.utc_string()
                benchmark_run.write_json(manifest_path, manifest)
                raise AblationError(str(error)) from error
        benchmark_run.write_json(manifest_path, manifest)

    manifest["status"] = "complete" if failures == 0 else "partial"
    manifest["failed_jobs"] = failures
    manifest["finished_at"] = benchmark_run.utc_string()
    benchmark_run.write_json(manifest_path, manifest)
    return manifest


def load_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "run_manifest.json"
    if not path.is_file():
        raise AblationError(f"Missing ablation manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AblationError(f"Cannot parse ablation manifest: {path}") from error
    if manifest.get("dataset") != "replica" or manifest.get("sequence") != SEQUENCE:
        raise AblationError(f"Not a Replica {SEQUENCE} ablation: {path}")
    return manifest


def manifest_variants(manifest: dict[str, Any]) -> tuple[Variant, ...]:
    keys = tuple(str(value["key"]) for value in manifest.get("variants", []))
    if not keys:
        raise AblationError("The ablation manifest has no variants")
    try:
        return tuple(VARIANTS[key] for key in keys)
    except KeyError as error:
        raise AblationError(
            f"Current script does not define manifest variant {error.args[0]}"
        ) from error


def complete_trials(manifest: dict[str, Any]) -> list[tuple[str, int, Path]]:
    result: list[tuple[str, int, Path]] = []
    for job in manifest.get("jobs", []):
        if job.get("status") != "complete":
            continue
        result.append(
            (
                str(job["variant"]),
                int(job["trial"]),
                Path(str(job["shutdown_dir"])),
            )
        )
    if not result:
        raise AblationError("The ablation contains no completed trials")
    return result


def evaluate_trial(
    args: argparse.Namespace,
    variant_key: str,
    trial: int,
    shutdown_dir: Path,
    images: list[Path],
    gt_trajectory: Path,
    gt_mesh: Path,
    output_dir: Path,
    lpips_runtime: tuple[Any, Any, Any] | None,
) -> dict[str, Any]:
    job = benchmark_evaluate.Job(
        method="ours",
        sequence=SEQUENCE,
        trial=trial,
        shutdown_dir=shutdown_dir,
    )
    values = benchmark_evaluate.metric_by_frame(job, images)
    frame_ids = sorted(values)
    lpips_value: float | None = None
    if lpips_runtime is not None:
        lpips_by_frame = benchmark_evaluate.lpips_for_frames(
            job,
            images,
            values,
            frame_ids,
            args.lpips_device,
            output_dir / "lpips_per_frame.json",
            lpips_runtime,
        )
        lpips_value = benchmark_evaluate.mean(
            lpips_by_frame[frame] for frame in frame_ids
        )

    geometry = benchmark_evaluate.evaluate_geometry(
        job,
        gt_mesh,
        gt_trajectory,
        output_dir / "geometry",
        args.distance_threshold_m,
        args.hislam_python,
        args.force_reconstruction,
    )
    row: dict[str, Any] = {
        "variant": variant_key,
        "trial": trial,
        "evaluated_frames": len(frame_ids),
        "accuracy_cm": 100.0 * geometry["mean precision"],
        "completeness_cm": 100.0 * geometry["mean recall"],
        "completion_ratio_percent": 100.0 * geometry["recall"],
        "precision_5cm": geometry["precision"],
        "recall_5cm": geometry["recall"],
        "fscore_5cm": geometry["f-score"],
        "psnr": benchmark_evaluate.mean(
            values[frame]["psnr"] for frame in frame_ids
        ),
        "ssim": benchmark_evaluate.mean(
            values[frame]["ssim"] for frame in frame_ids
        ),
        "lpips": lpips_value,
        "shutdown_dir": str(shutdown_dir),
        "raw_mesh": str(benchmark_evaluate.raw_mesh(job)),
        "gt_mesh": str(gt_mesh),
    }
    benchmark_run.write_json(output_dir / "metrics.json", row)
    print(
        f"[{variant_key}/trial_{trial:02d}] "
        f"Acc={row['accuracy_cm']:.3f} cm, "
        f"Comp={row['completeness_cm']:.3f} cm, "
        f"F@5={row['fscore_5cm']:.4f}, "
        f"PSNR={row['psnr']:.3f}, SSIM={row['ssim']:.4f}, "
        f"LPIPS={row['lpips'] if row['lpips'] is not None else '--'}",
        flush=True,
    )
    return row


def aggregate_trials(
    variants: Iterable[Variant], rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    result: dict[str, Any] = {}
    numeric_keys = tuple(METRIC_DIRECTIONS)
    for variant in variants:
        trials = grouped.get(variant.key, [])
        if not trials:
            continue
        means: dict[str, float | None] = {}
        standard_deviations: dict[str, float | None] = {}
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in trials
                if row.get(key) is not None
            ]
            means[key] = statistics.fmean(values) if values else None
            standard_deviations[key] = (
                statistics.pstdev(values) if values else None
            )
        result[variant.key] = {
            **variant_manifest(variant),
            "completed_trials": len(trials),
            "mean": means,
            "std": standard_deviations,
            "trials": sorted(trials, key=lambda row: int(row["trial"])),
        }
    return result


def best_by_metric(aggregated: dict[str, Any]) -> dict[str, Any]:
    winners: dict[str, Any] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        candidates = [
            (key, values["mean"][metric])
            for key, values in aggregated.items()
            if values["mean"].get(metric) is not None
            and math.isfinite(float(values["mean"][metric]))
        ]
        if not candidates:
            continue
        winner = (
            min(candidates, key=lambda item: item[1])
            if direction == "min"
            else max(candidates, key=lambda item: item[1])
        )
        winners[metric] = {
            "variant": winner[0],
            "value": winner[1],
            "direction": direction,
        }
    return winners


def print_summary(aggregated: dict[str, Any]) -> None:
    print("\n=== Replica office0 MVS-TSDF pruning ablation ===")
    print(
        f"{'Variant':27s} {'Acc cm':>8s} {'Comp cm':>8s} "
        f"{'F@5':>7s} {'PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s}"
    )
    ordered = sorted(
        aggregated.values(),
        key=lambda value: float(value["mean"]["fscore_5cm"]),
        reverse=True,
    )
    for value in ordered:
        mean = value["mean"]
        lpips = "--" if mean["lpips"] is None else f"{mean['lpips']:.4f}"
        print(
            f"{value['key']:27s} {mean['accuracy_cm']:8.3f} "
            f"{mean['completeness_cm']:8.3f} {mean['fscore_5cm']:7.4f} "
            f"{mean['psnr']:8.3f} {mean['ssim']:7.4f} {lpips:>7s}"
        )


def evaluate_ablation(
    args: argparse.Namespace,
    variants: tuple[Variant, ...],
    manifest: dict[str, Any],
    run_root: Path,
    data_dir: Path,
    gt_mesh: Path,
) -> dict[str, Any]:
    images = benchmark_evaluate.replica_images(data_dir)
    gt_trajectory = benchmark_evaluate.require_file(
        data_dir / "traj.txt", "Replica GT trajectory"
    )
    lpips_runtime = (
        None
        if args.skip_lpips
        else benchmark_evaluate.load_lpips(args.lpips_device)
    )
    rows: list[dict[str, Any]] = []
    for variant_key, trial, shutdown_dir in complete_trials(manifest):
        if variant_key not in {variant.key for variant in variants}:
            raise AblationError(
                f"Manifest contains unknown variant: {variant_key}"
            )
        output_dir = run_root / "evaluation" / variant_key / f"trial_{trial:02d}"
        rows.append(
            evaluate_trial(
                args,
                variant_key,
                trial,
                shutdown_dir,
                images,
                gt_trajectory,
                gt_mesh,
                output_dir,
                lpips_runtime,
            )
        )

    aggregated = aggregate_trials(variants, rows)
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": "replica",
        "sequence": SEQUENCE,
        "protocol": {
            "geometry": (
                "trajectory Sim(3) through mesh_eval, followed by HI-SLAM2 "
                "rigid ICP and 3D reconstruction evaluation"
            ),
            "distance_threshold_m": args.distance_threshold_m,
            "appearance": (
                "saved PSNR/SSIM and AlexNet LPIPS over every rendered "
                "keyframe in each trial"
            ),
            "trial_aggregation": "arithmetic mean and population standard deviation",
        },
        "variants": aggregated,
        "best_by_metric": best_by_metric(aggregated),
    }
    summary_path = run_root / "evaluation" / "summary.json"
    benchmark_run.write_json(summary_path, summary)
    print_summary(aggregated)
    print(f"\nSaved ablation summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and evaluate controlled pruning ablations around MVS-TSDF "
            "densification on Replica office0 without editing the canonical YAML."
        )
    )
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--suite", choices=tuple(SUITES), default="core")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        help="explicit variants; overrides --suite",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("PHOTOSLAM_DATA_ROOT", REPO_ROOT / "scripts/data")
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            os.environ.get("PHOTOSLAM_RESULTS_ROOT", REPO_ROOT / "results")
        ),
    )
    parser.add_argument(
        "--gt-mesh-root",
        type=Path,
        default=REPO_ROOT / "third_party/ESLAM/cull_replica_mesh",
    )
    parser.add_argument("--distance-threshold-m", type=float, default=0.05)
    parser.add_argument("--hislam-python", default=sys.executable)
    parser.add_argument("--lpips-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--force-reconstruction", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def print_variants() -> None:
    for suite, keys in SUITES.items():
        print(f"{suite}: {', '.join(keys)}")
    print("\nVariants:")
    for variant in VARIANTS.values():
        print(f"  {variant.key:27s} {variant.description}")


def main() -> int:
    args = parse_args()
    if args.list:
        print_variants()
        return 0
    if args.repetitions < 1:
        raise AblationError("--repetitions must be at least one")
    if not benchmark_run.SAFE_RUN_ID.fullmatch(args.run_id):
        raise AblationError("Invalid --run-id")
    if not math.isfinite(args.distance_threshold_m) or args.distance_threshold_m <= 0:
        raise AblationError("--distance-threshold-m must be positive")
    if args.evaluate_only and args.dry_run:
        raise AblationError("--evaluate-only and --dry-run are mutually exclusive")

    run_root = args.results_root / "densification_ablation" / args.run_id
    manifest: dict[str, Any] | None = None
    if args.evaluate_only:
        manifest = load_manifest(run_root)
        variants = manifest_variants(manifest)
    else:
        variants = selected_variants(args)
    validate_variant_matrix(variants)
    dataset = benchmark_run.DATASETS["replica"]
    data_dir = benchmark_run.resolve_data_dir(
        args.data_root, dataset, SEQUENCE
    )
    gt_mesh = args.gt_mesh_root / f"{SEQUENCE}.ply"
    benchmark_run.require_file(gt_mesh, "Replica office0 GT mesh")
    print(f"Run ID:       {args.run_id}")
    print(f"Dataset:      Replica/{SEQUENCE}")
    print(f"Data root:    {args.data_root}")
    print(f"Result root:  {run_root}")
    if args.evaluate_only:
        assert manifest is not None
        print("Mode:         evaluation only")
        print(
            "Variants:     "
            + ", ".join(variant.key for variant in variants)
        )
    else:
        benchmark_run.require_expected_hash(
            benchmark_run.VOCABULARY,
            benchmark_run.EXPECTED_VOCABULARY_SHA256,
            "ORB vocabulary",
        )
        if any(variant.requires_mvs for variant in variants):
            benchmark_run.require_expected_hash(
                benchmark_run.MVS_MODEL,
                benchmark_run.EXPECTED_MVS_MODEL_SHA256,
                "portable TANDEM MVS model",
            )
        jobs = make_jobs(variants, args.repetitions, data_dir, run_root)
        print(f"Suite:        {suite_name(args)}")
        print(f"Trials:       {len(jobs)}")
        for job in jobs:
            print(
                f"  {job.method.key:27s} trial {job.trial:02d} -> "
                f"{job.output_dir}"
            )
        if args.dry_run:
            print("Dry run complete; no outputs were created.")
            return 0
        if run_root.exists():
            raise AblationError(f"Refusing to overwrite ablation: {run_root}")
        require_evaluation_environment(args)
        manifest = run_ablation(args, variants, jobs, run_root)

    if args.evaluate_only:
        require_evaluation_environment(args)
    assert manifest is not None
    evaluate_ablation(args, variants, manifest, run_root, data_dir, gt_mesh)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AblationError, benchmark_run.BenchmarkError, benchmark_evaluate.EvaluationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
