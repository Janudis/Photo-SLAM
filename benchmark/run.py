#!/usr/bin/env python3
"""Run reproducible native Photo-SLAM paper benchmarks.

This launcher owns only experiment orchestration. It does not edit the
canonical dataset runners or mapper configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = REPO_ROOT / "ORB-SLAM3/Vocabulary/ORBvoc.txt"
MVS_MODEL = REPO_ROOT / "models/tandem/model.pt"
SHUTDOWN_PATTERN = re.compile(r"^\d+_shutdown$")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

EXPECTED_VOCABULARY_SHA256 = (
    "f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570"
)
EXPECTED_MVS_MODEL_SHA256 = (
    "145712c49cf9cd48d0bebefb3366603564de88675976a6a3c56bdbaba43bf2c5"
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    sequences: tuple[str, ...]
    data_candidates: tuple[str, ...]
    data_markers: tuple[str, ...]
    voxel_binary: str
    photoslam_binary: str
    voxel_config: str
    voxel_config_sha256: str
    orb_config: str
    photoslam_config: str


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    voxel_overrides: dict[str, int | float]
    requires_mvs_model: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "replica": DatasetSpec(
        key="replica",
        sequences=(
            "office0",
            "office1",
            "office2",
            "office3",
            "office4",
            "room0",
            "room1",
            "room2",
        ),
        data_candidates=("Replica/{sequence}", "{sequence}"),
        data_markers=("traj.txt", "results"),
        voxel_binary="bin/replica_mono_voxel",
        photoslam_binary="bin/replica_mono",
        voxel_config=(
            "cfg/voxel_mapper/Monocular/Replica/replica_mono_voxel.yaml"
        ),
        voxel_config_sha256=(
            "f6b42ee6cb9a61a119ff0f978d28f9409d91a828857ec449ee5dae07ef370611"
        ),
        orb_config="cfg/ORB_SLAM3/Monocular/Replica/{sequence}.yaml",
        photoslam_config=(
            "cfg/gaussian_mapper/Monocular/Replica/{sequence}.yaml"
        ),
    ),
    "tum": DatasetSpec(
        key="tum",
        sequences=(
            "rgbd_dataset_freiburg1_desk",
            "rgbd_dataset_freiburg2_xyz",
            "rgbd_dataset_freiburg3_long_office_household",
        ),
        data_candidates=("TUM/{sequence}", "{sequence}"),
        data_markers=("rgb.txt",),
        voxel_binary="bin/tum_mono_voxel",
        photoslam_binary="bin/tum_mono",
        voxel_config="cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml",
        voxel_config_sha256=(
            "9e9806a110f8e4c8b225a2e7bb2625ac5a635d18e0d43f57cc73d24f5ae6b599"
        ),
        orb_config="cfg/ORB_SLAM3/Monocular/TUM/{tum_config}.yaml",
        photoslam_config=(
            "cfg/gaussian_mapper/Monocular/TUM/{tum_config}.yaml"
        ),
    ),
    "scannet": DatasetSpec(
        key="scannet",
        sequences=("scene0000_00",),
        data_candidates=(
            "ScanNet/scans/{sequence}",
            "scans/{sequence}",
            "{sequence}",
        ),
        data_markers=("association.txt",),
        voxel_binary="bin/scannet_mono_voxel",
        photoslam_binary="bin/scannet_mono",
        voxel_config=(
            "cfg/voxel_mapper/Monocular/ScanNet/scannet_mono_voxel.yaml"
        ),
        voxel_config_sha256=(
            "d5b2d4b3788e5bbef3dde875248d63bc02a30d5126065369284ec6d408861c51"
        ),
        orb_config="cfg/ORB_SLAM3/Monocular/ScanNet/{sequence}.yaml",
        photoslam_config=(
            "cfg/gaussian_mapper/Monocular/ScanNet/scannet_mono.yaml"
        ),
    ),
}


_DISABLED_LEARNED_DEPTH_MODES = {
    "Mapper.monocular_mvs_tsdf_evidence": 0,
    "Mapper.monocular_omnidata_densify": 0,
    "Record.enable_rerun": 0,
}

_PHOTOSLAM_REPLICA_EXPORT_OVERRIDES: dict[str, int | float] = {
    "Record.save_rendered_mesh_eval": 1,
    "Record.rendered_mesh_eval_voxel_size_m": 0.03,
    "Record.rendered_mesh_eval_min_weight": 1.0,
    "Record.rendered_mesh_eval_trunc_vox": 8.0,
    "Record.rendered_mesh_eval_depth_max_m": 40.0,
    "Record.rendered_mesh_eval_alpha_thres": 0.5,
}

METHODS: dict[str, MethodSpec] = {
    "ours": MethodSpec(
        key="ours",
        label="Ours",
        family="voxel",
        voxel_overrides={
            "Mapper.monocular_rendered_depth_densify": 1,
            "Mapper.monocular_mvs_densify": 0,
            **_DISABLED_LEARNED_DEPTH_MODES,
        },
    ),
    "ours_mvs": MethodSpec(
        key="ours_mvs",
        label="Ours + MVS",
        family="voxel",
        voxel_overrides={
            "Mapper.monocular_rendered_depth_densify": 0,
            "Mapper.monocular_mvs_densify": 1,
            **_DISABLED_LEARNED_DEPTH_MODES,
        },
        requires_mvs_model=True,
    ),
    "photoslam": MethodSpec(
        key="photoslam",
        label="Photo-SLAM",
        family="photoslam",
        voxel_overrides={},
    ),
}


@dataclass(frozen=True)
class Job:
    dataset: DatasetSpec
    sequence: str
    method: MethodSpec
    trial: int
    data_dir: Path
    output_dir: Path
    orb_config: Path
    mapper_config: Path | None
    photoslam_config: Path | None
    binary: Path


class BenchmarkError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_string(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat(timespec="seconds")


def default_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise BenchmarkError(f"Missing {description}: {path}")


def require_expected_hash(
    path: Path, expected: str, description: str
) -> None:
    require_file(path, description)
    actual = sha256(path)
    if actual != expected:
        raise BenchmarkError(
            f"{description} changed from the fixed benchmark version: {path}\n"
            f"expected SHA-256: {expected}\n"
            f"actual SHA-256:   {actual}"
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def snapshot_file(source: Path, destination: Path) -> None:
    require_file(source, "configuration file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def apply_yaml_overrides(
    text: str,
    overrides: dict[str, int | float],
    *,
    append_missing: bool = False,
) -> str:
    result = text
    for key, value in overrides.items():
        pattern = re.compile(
            rf"^(?P<indent>[ \t]*){re.escape(key)}[ \t]*:[^\r\n]*$",
            re.MULTILINE,
        )
        result, count = pattern.subn(rf"\g<indent>{key}: {value}", result)
        if count == 0 and append_missing:
            if result and not result.endswith("\n"):
                result += "\n"
            result += f"{key}: {value}\n"
            continue
        if count != 1:
            raise BenchmarkError(
                f"Expected exactly one YAML key {key!r}; found {count}"
            )
    return result


def relative_config_path(template: str, sequence: str) -> Path:
    tum_config = sequence.removeprefix("rgbd_dataset_")
    return Path(
        template.format(sequence=sequence, tum_config=tum_config)
    )


def resolve_data_dir(
    data_root: Path, dataset: DatasetSpec, sequence: str
) -> Path:
    attempted: list[Path] = []
    for template in dataset.data_candidates:
        candidate = data_root / template.format(sequence=sequence)
        attempted.append(candidate)
        if not candidate.is_dir():
            continue
        if all((candidate / marker).exists() for marker in dataset.data_markers):
            return candidate
    attempted_text = "\n".join(f"  - {path}" for path in attempted)
    markers = ", ".join(dataset.data_markers)
    raise BenchmarkError(
        f"Cannot locate {dataset.key}/{sequence} below {data_root}.\n"
        f"Expected markers: {markers}\nTried:\n{attempted_text}"
    )


def expand_methods(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise BenchmarkError("Use --methods all by itself")
        return tuple(METHODS)
    unknown = sorted(set(values).difference(METHODS))
    if unknown:
        raise BenchmarkError(f"Unknown method(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(values))


def expand_sequences(dataset: DatasetSpec, values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise BenchmarkError("Use --sequences all by itself")
        return dataset.sequences
    unknown = sorted(set(values).difference(dataset.sequences))
    if unknown:
        raise BenchmarkError(
            f"Unsupported {dataset.key} sequence(s): {', '.join(unknown)}"
        )
    return tuple(dict.fromkeys(values))


def git_value(*arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def command_output(command: list[str]) -> str:
    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        return f"unavailable: {error}"
    return process.stdout.strip()


def machine_provenance() -> dict[str, Any]:
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as error:  # pragma: no cover - environment dependent
        torch_info = {"error": str(error)}

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_info,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def repository_provenance() -> dict[str, Any]:
    status = git_value("status", "--short")
    return {
        "root": str(REPO_ROOT),
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "submodules": git_value("submodule", "status").splitlines(),
    }


def build_jobs(
    dataset: DatasetSpec,
    sequences: Iterable[str],
    methods: Iterable[str],
    repetitions: int,
    data_root: Path,
    run_root: Path,
) -> list[Job]:
    require_expected_hash(
        VOCABULARY,
        EXPECTED_VOCABULARY_SHA256,
        "ORB vocabulary",
    )
    mapper_source = REPO_ROOT / dataset.voxel_config
    require_expected_hash(
        mapper_source,
        dataset.voxel_config_sha256,
        f"fixed {dataset.key} voxel configuration",
    )

    jobs: list[Job] = []
    for sequence in sequences:
        data_dir = resolve_data_dir(data_root, dataset, sequence)
        orb_config = REPO_ROOT / relative_config_path(
            dataset.orb_config, sequence
        )
        require_file(orb_config, "ORB-SLAM configuration")
        for method_key in methods:
            method = METHODS[method_key]
            if method.requires_mvs_model:
                require_expected_hash(
                    MVS_MODEL,
                    EXPECTED_MVS_MODEL_SHA256,
                    "portable TANDEM MVS model",
                )
            if method.family == "voxel":
                binary = REPO_ROOT / dataset.voxel_binary
                photoslam_config = None
            else:
                binary = REPO_ROOT / dataset.photoslam_binary
                photoslam_config = REPO_ROOT / relative_config_path(
                    dataset.photoslam_config, sequence
                )
                require_file(
                    photoslam_config, "Photo-SLAM mapper configuration"
                )
            require_file(binary, f"{method.label} executable")
            if not os.access(binary, os.X_OK):
                raise BenchmarkError(f"Executable is not runnable: {binary}")
            for trial in range(1, repetitions + 1):
                output_dir = (
                    run_root
                    / dataset.key
                    / sequence
                    / method.key
                    / f"trial_{trial:02d}"
                )
                if output_dir.exists():
                    raise BenchmarkError(
                        f"Refusing to overwrite benchmark output: {output_dir}"
                    )
                jobs.append(
                    Job(
                        dataset=dataset,
                        sequence=sequence,
                        method=method,
                        trial=trial,
                        data_dir=data_dir,
                        output_dir=output_dir,
                        orb_config=orb_config,
                        mapper_config=(
                            mapper_source
                            if method.family == "voxel"
                            else None
                        ),
                        photoslam_config=photoslam_config,
                        binary=binary,
                    )
                )
    return jobs


def prepare_job(job: Job) -> tuple[list[str], dict[str, str]]:
    config_dir = job.output_dir / "configs"
    orb_snapshot = config_dir / "orb_slam.yaml"
    snapshot_file(job.orb_config, orb_snapshot)

    config_hashes = {"orb_slam.yaml": sha256(orb_snapshot)}
    if job.method.family == "voxel":
        assert job.mapper_config is not None
        mapper_text = job.mapper_config.read_text(encoding="utf-8")
        effective_text = apply_yaml_overrides(
            mapper_text, job.method.voxel_overrides
        )
        mapper_snapshot = config_dir / "voxel_mapper.yaml"
        mapper_snapshot.write_text(effective_text, encoding="utf-8")
        config_hashes["voxel_mapper.yaml"] = sha256(mapper_snapshot)
        command = [
            str(job.binary),
            str(VOCABULARY),
            str(orb_snapshot),
            str(mapper_snapshot),
            str(job.data_dir),
            str(job.output_dir),
            "no_viewer",
        ]
    else:
        assert job.photoslam_config is not None
        mapper_snapshot = config_dir / "gaussian_mapper.yaml"
        mapper_text = job.photoslam_config.read_text(encoding="utf-8")
        if job.dataset.key == "replica":
            mapper_text = apply_yaml_overrides(
                mapper_text,
                _PHOTOSLAM_REPLICA_EXPORT_OVERRIDES,
                append_missing=True,
            )
        mapper_snapshot.parent.mkdir(parents=True, exist_ok=True)
        mapper_snapshot.write_text(mapper_text, encoding="utf-8")
        config_hashes["gaussian_mapper.yaml"] = sha256(mapper_snapshot)
        command = [
            str(job.binary),
            str(VOCABULARY),
            str(orb_snapshot),
            str(mapper_snapshot),
            str(job.data_dir),
            str(job.output_dir),
            "no_viewer",
        ]
    return command, config_hashes


def stream_process(command: list[str], log_path: Path) -> int:
    environment = os.environ.copy()
    simple_knn = str(REPO_ROOT / "third_party/simple-knn")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{simple_knn}:{existing_pythonpath}"
        if existing_pythonpath
        else simple_knn
    )

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def discover_shutdown(output_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_dir() and SHUTDOWN_PATTERN.fullmatch(path.name)
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise BenchmarkError(
            f"Expected one numeric *_shutdown directory in {output_dir}; "
            f"found {names}"
        )
    return candidates[0]


def run_job(
    job: Job,
    run_id: str,
    repository: dict[str, Any],
    machine: dict[str, Any],
) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True)
    command, config_hashes = prepare_job(job)
    manifest_path = job.output_dir / "provenance.json"
    started = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "method": job.method.key,
        "method_label": job.method.label,
        "dataset": job.dataset.key,
        "sequence": job.sequence,
        "trial": job.trial,
        "started_at": utc_string(started),
        "command": command,
        "command_shell": shlex.join(command),
        "paths": {
            "data": str(job.data_dir),
            "output": str(job.output_dir),
            "log": str(job.output_dir / "run.log"),
        },
        "config_sha256": config_hashes,
        "model_sha256": (
            EXPECTED_MVS_MODEL_SHA256
            if job.method.requires_mvs_model
            else None
        ),
        "vocabulary_sha256": EXPECTED_VOCABULARY_SHA256,
        "repository": repository,
        "machine": machine,
    }
    write_json(manifest_path, manifest)

    print(
        f"\n=== {job.method.label} | {job.dataset.key}/{job.sequence} "
        f"| trial {job.trial:02d} ===",
        flush=True,
    )
    print(f"$ {shlex.join(command)}", flush=True)
    try:
        return_code = stream_process(command, job.output_dir / "run.log")
        if return_code != 0:
            raise BenchmarkError(
                f"Process exited with status {return_code}; see "
                f"{job.output_dir / 'run.log'}"
            )
        shutdown = discover_shutdown(job.output_dir)
    except BaseException as error:
        finished = utc_now()
        manifest.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "finished_at": utc_string(finished),
                "elapsed_seconds": (finished - started).total_seconds(),
                "error": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise

    finished = utc_now()
    manifest.update(
        {
            "status": "complete",
            "finished_at": utc_string(finished),
            "elapsed_seconds": (finished - started).total_seconds(),
            "shutdown_dir": str(shutdown),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated Ours, Ours+MVS, and original Photo-SLAM "
            "benchmarks with fixed method presets."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=tuple(DATASETS),
        help="dataset family to run",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        metavar="METHOD",
        help="ours, ours_mvs, photoslam, or all (default: all)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["all"],
        metavar="SEQUENCE",
        help="one or more supported sequences, or all (default: all)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="independent runs per method and sequence (default: 1)",
    )
    parser.add_argument(
        "--run-id",
        default=default_run_id(),
        help="unique output group name (default: UTC timestamp)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("PHOTOSLAM_DATA_ROOT", REPO_ROOT / "scripts/data")
        ),
        help="dataset root (default: PHOTOSLAM_DATA_ROOT or scripts/data)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            os.environ.get("PHOTOSLAM_RESULTS_ROOT", REPO_ROOT / "results")
        ),
        help="result root (default: PHOTOSLAM_RESULTS_ROOT or results)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue later jobs after a failed experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the job matrix without creating outputs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list native benchmark coverage and exit",
    )
    return parser.parse_args()


def print_coverage() -> None:
    print("Native methods:")
    for method in METHODS.values():
        print(f"  {method.key:10s} {method.label}")
    print("Datasets:")
    for dataset in DATASETS.values():
        print(f"  {dataset.key:10s} {', '.join(dataset.sequences)}")


def main() -> int:
    args = parse_args()
    if args.list:
        print_coverage()
        return 0
    if args.dataset is None:
        raise BenchmarkError("dataset is required unless --list is used")
    if args.repetitions < 1:
        raise BenchmarkError("--repetitions must be at least 1")
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise BenchmarkError(
            "--run-id may contain only letters, digits, '.', '_' and '-'"
        )

    dataset = DATASETS[args.dataset]
    method_keys = expand_methods(args.methods)
    sequences = expand_sequences(dataset, args.sequences)
    run_root = args.results_root / "paper_benchmark" / args.run_id
    if run_root.exists():
        raise BenchmarkError(
            f"Refusing to reuse benchmark run directory: {run_root}"
        )

    jobs = build_jobs(
        dataset=dataset,
        sequences=sequences,
        methods=method_keys,
        repetitions=args.repetitions,
        data_root=args.data_root,
        run_root=run_root,
    )
    print(f"Run ID:       {args.run_id}")
    print(f"Dataset root: {args.data_root}")
    print(f"Result root:  {run_root}")
    print(f"Jobs:         {len(jobs)}")
    for job in jobs:
        print(
            f"  {job.dataset.key}/{job.sequence} | {job.method.key} | "
            f"trial {job.trial:02d} -> {job.output_dir}"
        )
    if args.dry_run:
        print("Dry run complete; no output was created.")
        return 0

    repository = repository_provenance()
    machine = machine_provenance()
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "created_at": utc_string(),
        "dataset": dataset.key,
        "sequences": list(sequences),
        "methods": list(method_keys),
        "repetitions": args.repetitions,
        "repository": repository,
        "machine": machine,
        "jobs": [],
    }
    write_json(run_manifest_path, run_manifest)

    failed = 0
    for job in jobs:
        try:
            result = run_job(job, args.run_id, repository, machine)
            run_manifest["jobs"].append(
                {
                    "status": "complete",
                    "method": job.method.key,
                    "sequence": job.sequence,
                    "trial": job.trial,
                    "provenance": str(job.output_dir / "provenance.json"),
                    "shutdown_dir": result["shutdown_dir"],
                }
            )
        except KeyboardInterrupt:
            run_manifest["status"] = "interrupted"
            run_manifest["finished_at"] = utc_string()
            write_json(run_manifest_path, run_manifest)
            raise
        except Exception as error:
            failed += 1
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            run_manifest["jobs"].append(
                {
                    "status": "failed",
                    "method": job.method.key,
                    "sequence": job.sequence,
                    "trial": job.trial,
                    "provenance": str(job.output_dir / "provenance.json"),
                    "error": str(error),
                }
            )
            write_json(run_manifest_path, run_manifest)
            if not args.continue_on_error:
                run_manifest["status"] = "failed"
                run_manifest["finished_at"] = utc_string()
                write_json(run_manifest_path, run_manifest)
                return 1
        write_json(run_manifest_path, run_manifest)

    run_manifest["status"] = "complete" if failed == 0 else "partial"
    run_manifest["finished_at"] = utc_string()
    run_manifest["failed_jobs"] = failed
    write_json(run_manifest_path, run_manifest)
    print(f"\nBenchmark manifest: {run_manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
