#!/usr/bin/env python3
"""Evaluate surface normals derived from rendered metric depth maps.

All methods and ground truth pass through the same projective depth-to-normal
implementation. Inputs must be single-channel metric depth images; colorized or
8-bit visualization images are rejected.

Example (Replica):
  python3 evaluation/normal_maps_evaluation.py \
    --camera-yaml cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    --gt-depth-root scripts/data/Replica/office0 \
    --gt-depth-template 'results/depth{frame_id:06d}.png' \
    --method Ours results/.../depth_metric results/.../kf_frame_id_map.txt 1000 \
    --method Photo-SLAM results/.../depth_metric results/.../ply/cameras.json 1000 \
    --method HI-SLAM2 third_party/HI-SLAM2/outputs/.../renders/depth_after_opt - 6553.5 \
    --out results/.../normal_evaluation

For TUM, replace --gt-depth-template with --gt-associations and point
--gt-depth-root at the sequence directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


ANGLE_THRESHOLDS = (11.25, 22.5, 30.0)
HISTOGRAM_STEP_DEG = 0.01
HISTOGRAM_EDGES = np.arange(
    0.0, 180.0 + HISTOGRAM_STEP_DEG, HISTOGRAM_STEP_DEG, dtype=np.float64
)


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float


@dataclass(frozen=True)
class MethodInput:
    name: str
    depth_dir: Path
    id_map_path: Optional[Path]
    depth_scale: float
    frames: Mapping[int, Path]


class GroundTruthDepth:
    def __init__(
        self,
        root: Path,
        template: Optional[str],
        associations: Optional[Path],
    ) -> None:
        self.root = root
        self.template = template
        self.by_frame: Dict[int, Path] = {}
        self.rgb_name_to_frame: Dict[str, int] = {}
        if associations is not None:
            self._load_associations(associations)

    def _load_associations(self, path: Path) -> None:
        frame_id = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed association line in {path}: {line}")
                rgb_path = fields[1]
                depth_path = fields[3]
                self.by_frame[frame_id] = self.root / depth_path
                self.rgb_name_to_frame[Path(rgb_path).name] = frame_id
                self.rgb_name_to_frame[Path(rgb_path).stem] = frame_id
                self.rgb_name_to_frame[fields[0]] = frame_id
                frame_id += 1

    def path_for(self, frame_id: int) -> Path:
        if self.template is not None:
            return self.root / self.template.format(frame_id=frame_id)
        try:
            return self.by_frame[frame_id]
        except KeyError as exc:
            raise KeyError(f"No GT depth path for dataset frame {frame_id}") from exc

    def frame_ids(self) -> Iterable[int]:
        if self.template is not None:
            return ()
        return self.by_frame.keys()

    def frame_from_image_name(self, image_name: str) -> Optional[int]:
        path = Path(image_name)
        for key in (path.name, path.stem):
            if key in self.rgb_name_to_frame:
                return self.rgb_name_to_frame[key]
        match = re.search(r"(?:frame|depth)(\d+)$", path.stem)
        if match:
            return int(match.group(1))
        return None


class StreamingAngularStats:
    def __init__(self) -> None:
        self.frames = 0
        self.gt_normals = 0
        self.predicted_normals_on_gt = 0
        self.overlap_normals = 0
        self.angle_sum = 0.0
        self.angle_sq_sum = 0.0
        self.threshold_hits = {threshold: 0 for threshold in ANGLE_THRESHOLDS}
        self.histogram = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)

    def update(self, errors_deg: np.ndarray, gt_count: int, predicted_count: int) -> None:
        errors = np.asarray(errors_deg, dtype=np.float64)
        self.frames += 1
        self.gt_normals += int(gt_count)
        self.predicted_normals_on_gt += int(predicted_count)
        self.overlap_normals += int(errors.size)
        if errors.size == 0:
            return
        self.angle_sum += float(errors.sum(dtype=np.float64))
        self.angle_sq_sum += float(np.square(errors).sum(dtype=np.float64))
        for threshold in ANGLE_THRESHOLDS:
            self.threshold_hits[threshold] += int(np.count_nonzero(errors < threshold))
        self.histogram += np.histogram(errors, bins=HISTOGRAM_EDGES)[0]

    def median(self) -> float:
        if self.overlap_normals == 0:
            return math.nan
        target = (self.overlap_normals + 1) // 2
        index = int(np.searchsorted(np.cumsum(self.histogram), target, side="left"))
        index = min(index, len(self.histogram) - 1)
        return float((HISTOGRAM_EDGES[index] + HISTOGRAM_EDGES[index + 1]) * 0.5)

    def metrics(self) -> Dict[str, object]:
        overlap = self.overlap_normals
        gt = self.gt_normals
        mean = self.angle_sum / overlap if overlap else math.nan
        rmse = math.sqrt(self.angle_sq_sum / overlap) if overlap else math.nan
        result: Dict[str, object] = {
            "evaluated_frames": self.frames,
            "valid_gt_normals": gt,
            "valid_predicted_normals_on_gt": self.predicted_normals_on_gt,
            "overlap_normals": overlap,
            "mean_angular_error_deg": mean,
            "median_angular_error_deg": self.median(),
            "rmse_angular_error_deg": rmse,
            "normal_coverage": self.predicted_normals_on_gt / gt if gt else math.nan,
        }
        for threshold in ANGLE_THRESHOLDS:
            label = format_threshold(threshold)
            hits = self.threshold_hits[threshold]
            result[f"accuracy_below_{label}_overlap"] = hits / overlap if overlap else math.nan
            result[f"accuracy_below_{label}_all_gt"] = hits / gt if gt else math.nan
        return result


def format_threshold(value: float) -> str:
    return f"{value:g}deg".replace(".", "p")


def finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def read_camera(path: Path) -> Camera:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Cannot open camera YAML: {path}")

    def value(*names: str) -> float:
        for name in names:
            node = storage.getNode(name)
            if not node.empty():
                return float(node.real())
        raise KeyError(f"None of {names} exists in {path}")

    camera = Camera(
        fx=value("Camera1.fx", "Camera.fx"),
        fy=value("Camera1.fy", "Camera.fy"),
        cx=value("Camera1.cx", "Camera.cx"),
        cy=value("Camera1.cy", "Camera.cy"),
        width=int(round(value("Camera.width", "Camera1.width"))),
        height=int(round(value("Camera.height", "Camera1.height"))),
        depth_scale=value("RGBD.DepthMapFactor", "DepthMapFactor"),
    )
    storage.release()
    return camera


def parse_text_id_map(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed ID-map line in {path}: {line}")
            mapping[int(fields[0])] = int(fields[1])
    if not mapping:
        raise ValueError(f"ID map contains no entries: {path}")
    return mapping


def parse_json_id_map(path: Path, gt: GroundTruthDepth) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Camera JSON must contain a list: {path}")
    mapping: Dict[int, int] = {}
    for record in records:
        if not isinstance(record, dict) or "id" not in record or "img_name" not in record:
            continue
        frame_id = gt.frame_from_image_name(str(record["img_name"]))
        if frame_id is None:
            raise ValueError(
                f"Cannot map camera image '{record['img_name']}' to a dataset frame. "
                "Use an explicit two-column ID map."
            )
        mapping[int(record["id"])] = frame_id
    if not mapping:
        raise ValueError(f"Camera JSON contains no usable entries: {path}")
    return mapping


def parse_local_depth_id(path: Path) -> Optional[int]:
    stem = path.stem
    match = re.fullmatch(r"kf_(\d+)", stem)
    if match:
        return int(match.group(1))
    if stem.isdigit():
        return int(stem)
    return None


def discover_method(
    values: Sequence[str],
    gt: GroundTruthDepth,
) -> MethodInput:
    name, depth_dir_text, id_map_text, depth_scale_text = values
    depth_dir = Path(depth_dir_text)
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory for {name} does not exist: {depth_dir}")
    depth_scale = float(depth_scale_text)
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError(f"Depth scale for {name} must be positive")

    id_map_path: Optional[Path]
    if id_map_text == "-":
        id_map_path = None
        local_to_dataset: Optional[Mapping[int, int]] = None
    else:
        id_map_path = Path(id_map_text)
        if not id_map_path.is_file():
            raise FileNotFoundError(f"ID map for {name} does not exist: {id_map_path}")
        if id_map_path.suffix.lower() == ".json":
            local_to_dataset = parse_json_id_map(id_map_path, gt)
        else:
            local_to_dataset = parse_text_id_map(id_map_path)

    frames: Dict[int, Path] = {}
    recognized = 0
    for path in sorted(depth_dir.glob("*.png")):
        local_id = parse_local_depth_id(path)
        if local_id is None:
            continue
        recognized += 1
        if local_to_dataset is None:
            dataset_id = local_id
        elif local_id in local_to_dataset:
            dataset_id = local_to_dataset[local_id]
        else:
            continue
        if dataset_id in frames:
            raise ValueError(
                f"Multiple depth files for {name} map to dataset frame {dataset_id}: "
                f"{frames[dataset_id]} and {path}"
            )
        frames[dataset_id] = path

    if recognized == 0:
        raise ValueError(
            f"No metric depth PNGs with names kf_XXXXX.png or XXXXXX.png found in {depth_dir}"
        )
    if not frames:
        raise ValueError(f"No {name} depth files matched its ID map")
    return MethodInput(name, depth_dir, id_map_path, depth_scale, frames)


def read_metric_depth(path: Path, scale: float, label: str) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Failed to read {label} depth: {path}")
    if raw.ndim != 2:
        raise ValueError(
            f"{label} depth must be single-channel metric data, got shape {raw.shape}: {path}"
        )
    if raw.dtype == np.uint8:
        raise ValueError(
            f"{label} depth is 8-bit and is probably a visualization, not metric depth: {path}"
        )
    if np.issubdtype(raw.dtype, np.integer):
        depth = raw.astype(np.float32) / np.float32(scale)
    elif np.issubdtype(raw.dtype, np.floating):
        depth = raw.astype(np.float32) / np.float32(scale)
    else:
        raise ValueError(f"Unsupported {label} depth dtype {raw.dtype}: {path}")
    depth[~np.isfinite(depth)] = 0.0
    return depth


def resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape == (height, width):
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


def depth_to_normals(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float,
    max_depth: float,
    discontinuity_abs_m: float,
    discontinuity_rel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    valid_depth = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    points = np.stack(
        ((u - cx) * depth / fx, (v - cy) * depth / fy, depth),
        axis=-1,
    )

    normals = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    if height < 3 or width < 3:
        return normals, mask

    center = depth[1:-1, 1:-1]
    left = depth[1:-1, :-2]
    right = depth[1:-1, 2:]
    up = depth[:-2, 1:-1]
    down = depth[2:, 1:-1]
    support = (
        valid_depth[1:-1, 1:-1]
        & valid_depth[1:-1, :-2]
        & valid_depth[1:-1, 2:]
        & valid_depth[:-2, 1:-1]
        & valid_depth[2:, 1:-1]
    )
    tolerance = np.maximum(discontinuity_abs_m, discontinuity_rel * center)
    continuous = (
        (np.abs(left - center) <= tolerance)
        & (np.abs(right - center) <= tolerance)
        & (np.abs(up - center) <= tolerance)
        & (np.abs(down - center) <= tolerance)
    )

    tangent_x = points[1:-1, 2:] - points[1:-1, :-2]
    tangent_y = points[2:, 1:-1] - points[:-2, 1:-1]
    normal = np.cross(tangent_y, tangent_x)
    magnitude = np.linalg.norm(normal, axis=-1)
    interior_valid = support & continuous & np.isfinite(magnitude) & (magnitude > 1e-12)
    normal[interior_valid] /= magnitude[interior_valid, None]

    center_points = points[1:-1, 1:-1]
    points_away = np.sum(normal * center_points, axis=-1) > 0.0
    normal[points_away] *= -1.0
    normal[~interior_valid] = 0.0
    normals[1:-1, 1:-1] = normal
    mask[1:-1, 1:-1] = interior_valid
    return normals, mask


def frame_metrics(
    gt_normals: np.ndarray,
    gt_valid: np.ndarray,
    pred_normals: np.ndarray,
    pred_valid: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    pred_on_gt = pred_valid & gt_valid
    overlap = pred_on_gt
    dots = np.sum(gt_normals[overlap] * pred_normals[overlap], axis=-1)
    errors = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))).astype(np.float32)
    gt_count = int(np.count_nonzero(gt_valid))
    pred_count = int(np.count_nonzero(pred_on_gt))
    overlap_count = int(errors.size)

    result: Dict[str, object] = {
        "valid_gt_normals": gt_count,
        "valid_predicted_normals_on_gt": pred_count,
        "overlap_normals": overlap_count,
        "normal_coverage": pred_count / gt_count if gt_count else math.nan,
        "mean_angular_error_deg": float(np.mean(errors)) if overlap_count else math.nan,
        "median_angular_error_deg": float(np.median(errors)) if overlap_count else math.nan,
        "rmse_angular_error_deg": (
            float(np.sqrt(np.mean(np.square(errors, dtype=np.float64))))
            if overlap_count
            else math.nan
        ),
    }
    for threshold in ANGLE_THRESHOLDS:
        label = format_threshold(threshold)
        hits = int(np.count_nonzero(errors < threshold))
        result[f"accuracy_below_{label}_overlap"] = (
            hits / overlap_count if overlap_count else math.nan
        )
        result[f"accuracy_below_{label}_all_gt"] = hits / gt_count if gt_count else math.nan
    return errors, result


def save_evaluated_normal_map(
    normal: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
) -> None:
    normal_rgb = np.clip((normal + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    normal_rgb[~valid] = 0
    normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), normal_bgr):
        raise RuntimeError(f"Failed to save evaluated normal map: {output_path}")


def choose_resolution(
    args: argparse.Namespace,
    common_frames: Sequence[int],
    gt: GroundTruthDepth,
    gt_scale: float,
    methods: Sequence[MethodInput],
) -> Tuple[int, int]:
    if (args.eval_width is None) != (args.eval_height is None):
        raise ValueError("--eval-width and --eval-height must be supplied together")
    if args.eval_width is not None:
        return int(args.eval_width), int(args.eval_height)

    frame_id = common_frames[0]
    candidates: List[Tuple[int, int]] = []
    gt_depth = read_metric_depth(gt.path_for(frame_id), gt_scale, "GT")
    candidates.append((gt_depth.shape[1], gt_depth.shape[0]))
    for method in methods:
        depth = read_metric_depth(
            method.frames[frame_id], method.depth_scale, f"{method.name} predicted"
        )
        candidates.append((depth.shape[1], depth.shape[0]))
    return min(candidates, key=lambda shape: shape[0] * shape[1])


def write_text_report(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "Normal-map evaluation from common metric-depth normals",
        f"evaluated frames: {report['evaluated_frame_count']}",
        f"evaluation resolution: {report['evaluation_width']}x{report['evaluation_height']}",
        f"depth discontinuity: abs={report['discontinuity_abs_m']} m, rel={report['discontinuity_rel']}",
        "",
    ]
    header = (
        f"{'Method':<22} {'Mean':>8} {'Median':>8} {'RMSE':>8} "
        f"{'<11.25':>9} {'<22.5':>9} {'<30':>9} {'Coverage':>10} "
        f"{'AllGT<11.25':>12} {'AllGT<22.5':>11} {'AllGT<30':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, metrics in report["methods"].items():
        def fmt(key: str, percent: bool = False) -> str:
            value = metrics[key]
            if value is None:
                return "N/A"
            return f"{100.0 * value:.2f}%" if percent else f"{value:.2f}"

        lines.append(
            f"{name:<22} "
            f"{fmt('mean_angular_error_deg'):>8} "
            f"{fmt('median_angular_error_deg'):>8} "
            f"{fmt('rmse_angular_error_deg'):>8} "
            f"{fmt('accuracy_below_11p25deg_overlap', True):>9} "
            f"{fmt('accuracy_below_22p5deg_overlap', True):>9} "
            f"{fmt('accuracy_below_30deg_overlap', True):>9} "
            f"{fmt('normal_coverage', True):>10} "
            f"{fmt('accuracy_below_11p25deg_all_gt', True):>12} "
            f"{fmt('accuracy_below_22p5deg_all_gt', True):>11} "
            f"{fmt('accuracy_below_30deg_all_gt', True):>9}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cdf(
    out_dir: Path,
    stats_by_method: Mapping[str, StreamingAngularStats],
) -> None:
    angle = HISTOGRAM_EDGES[1:]
    cdfs: Dict[str, np.ndarray] = {}
    for name, stats in stats_by_method.items():
        if stats.overlap_normals:
            cdfs[name] = np.cumsum(stats.histogram) / float(stats.overlap_normals)
        else:
            cdfs[name] = np.zeros_like(angle)

    with (out_dir / "normal_angular_error_cdf.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        names = list(cdfs)
        writer.writerow(["angle_deg", *names])
        for index, x_value in enumerate(angle):
            writer.writerow([f"{x_value:.2f}", *[f"{cdfs[name][index]:.8f}" for name in names]])

    width, height = 1350, 900
    left, right, top, bottom = 130, 50, 60, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_pixel(x_value: float, y_value: float) -> Tuple[int, int]:
        x = left + int(round(np.clip(x_value / 90.0, 0.0, 1.0) * plot_width))
        y = top + plot_height - int(round(np.clip(y_value, 0.0, 1.0) * plot_height))
        return x, y

    for x_tick in range(0, 91, 15):
        x, _ = to_pixel(float(x_tick), 0.0)
        cv2.line(canvas, (x, top), (x, top + plot_height), (225, 225, 225), 1)
        cv2.putText(
            canvas, str(x_tick), (x - 13, top + plot_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA,
        )
    for y_tick in np.linspace(0.0, 1.0, 6):
        _, y = to_pixel(0.0, float(y_tick))
        cv2.line(canvas, (left, y), (left + plot_width, y), (225, 225, 225), 1)
        cv2.putText(
            canvas, f"{y_tick:.1f}", (55, y + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA,
        )
    cv2.rectangle(
        canvas, (left, top), (left + plot_width, top + plot_height), (40, 40, 40), 2
    )
    cv2.putText(
        canvas, "Angular error (degrees)",
        (left + plot_width // 2 - 130, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "CDF", (20, top + plot_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA,
    )

    colors = [
        (31, 119, 180),
        (14, 127, 255),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]
    visible = angle <= 90.0
    legend_x = left + 25
    legend_y = top + 35
    for method_index, (name, cdf) in enumerate(cdfs.items()):
        color = colors[method_index % len(colors)]
        points = np.asarray(
            [to_pixel(float(x_value), float(y_value)) for x_value, y_value in zip(angle[visible], cdf[visible])],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
        y = legend_y + 32 * method_index
        cv2.line(canvas, (legend_x, y), (legend_x + 42, y), color, 4, cv2.LINE_AA)
        cv2.putText(
            canvas, name, (legend_x + 55, y + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "normal_angular_error_cdf.png"), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-yaml", type=Path, required=True)
    parser.add_argument("--gt-depth-root", type=Path, required=True)
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "--gt-depth-template",
        help="Relative template containing {frame_id}, e.g. results/depth{frame_id:06d}.png",
    )
    gt_group.add_argument("--gt-associations", type=Path)
    parser.add_argument(
        "--gt-depth-scale",
        type=float,
        default=None,
        help="Stored units per meter; defaults to RGBD.DepthMapFactor from camera YAML",
    )
    parser.add_argument(
        "--method",
        nargs=4,
        action="append",
        metavar=("NAME", "DEPTH_DIR", "ID_MAP", "DEPTH_SCALE"),
        required=True,
        help="Repeat per method. Use '-' for ID_MAP when file IDs are dataset frame IDs.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-width", type=int)
    parser.add_argument("--eval-height", type=int)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--discontinuity-abs-m", type=float, default=0.05)
    parser.add_argument("--discontinuity-rel", type=float, default=0.02)
    parser.add_argument(
        "--save-normal-maps",
        action="append",
        default=[],
        metavar="METHOD",
        help=(
            "Save the evaluated camera-space normal PNGs for this method under "
            "OUT/normal_maps/METHOD. Repeat for multiple methods."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = read_camera(args.camera_yaml)
    gt_scale = args.gt_depth_scale if args.gt_depth_scale is not None else camera.depth_scale
    gt = GroundTruthDepth(args.gt_depth_root, args.gt_depth_template, args.gt_associations)
    methods = [discover_method(values, gt) for values in args.method]
    names = [method.name for method in methods]
    if len(set(names)) != len(names):
        raise ValueError("Method names must be unique")
    unknown_normal_outputs = sorted(set(args.save_normal_maps) - set(names))
    if unknown_normal_outputs:
        raise ValueError(
            "--save-normal-maps names are not configured methods: "
            + ", ".join(unknown_normal_outputs)
        )

    common = set(methods[0].frames)
    for method in methods[1:]:
        common &= set(method.frames)
    if args.gt_associations is not None:
        common &= set(gt.frame_ids())
    missing_gt_frames = sorted(frame_id for frame_id in common if not gt.path_for(frame_id).is_file())
    common_frames = sorted(frame_id for frame_id in common if gt.path_for(frame_id).is_file())
    if not common_frames:
        raise RuntimeError("No exact dataset frame IDs are shared by every method and GT")
    if missing_gt_frames:
        print(
            f"Excluded {len(missing_gt_frames)} shared method frames with no exact GT depth file"
        )

    eval_width, eval_height = choose_resolution(args, common_frames, gt, gt_scale, methods)
    if eval_width <= 2 or eval_height <= 2:
        raise ValueError("Evaluation resolution must be at least 3x3")
    fx = camera.fx * eval_width / camera.width
    fy = camera.fy * eval_height / camera.height
    cx = camera.cx * eval_width / camera.width
    cy = camera.cy * eval_height / camera.height

    stats_by_method = {method.name: StreamingAngularStats() for method in methods}
    per_frame_rows: List[Dict[str, object]] = []
    for frame_id in common_frames:
        gt_depth = resize_depth(
            read_metric_depth(gt.path_for(frame_id), gt_scale, "GT"),
            eval_width,
            eval_height,
        )
        gt_normals, gt_valid = depth_to_normals(
            gt_depth,
            fx,
            fy,
            cx,
            cy,
            args.min_depth,
            args.max_depth,
            args.discontinuity_abs_m,
            args.discontinuity_rel,
        )
        gt_count = int(np.count_nonzero(gt_valid))

        for method in methods:
            pred_depth = resize_depth(
                read_metric_depth(
                    method.frames[frame_id],
                    method.depth_scale,
                    f"{method.name} predicted",
                ),
                eval_width,
                eval_height,
            )
            pred_normals, pred_valid = depth_to_normals(
                pred_depth,
                fx,
                fy,
                cx,
                cy,
                args.min_depth,
                args.max_depth,
                args.discontinuity_abs_m,
                args.discontinuity_rel,
            )
            if method.name in args.save_normal_maps:
                save_evaluated_normal_map(
                    pred_normals,
                    pred_valid,
                    args.out / "normal_maps" / method.name / f"frame_{frame_id:06d}.png",
                )
            errors, metrics = frame_metrics(gt_normals, gt_valid, pred_normals, pred_valid)
            stats_by_method[method.name].update(
                errors,
                gt_count,
                int(metrics["valid_predicted_normals_on_gt"]),
            )
            per_frame_rows.append(
                {"method": method.name, "frame_id": frame_id, **metrics}
            )

    report: Dict[str, object] = {
        "protocol": "common_projective_depth_to_normal",
        "camera_yaml": str(args.camera_yaml),
        "gt_depth_root": str(args.gt_depth_root),
        "gt_depth_scale": gt_scale,
        "evaluation_width": eval_width,
        "evaluation_height": eval_height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "discontinuity_abs_m": args.discontinuity_abs_m,
        "discontinuity_rel": args.discontinuity_rel,
        "evaluated_frame_count": len(common_frames),
        "evaluated_frame_ids": common_frames,
        "missing_gt_frame_ids": missing_gt_frames,
        "saved_normal_map_methods": sorted(set(args.save_normal_maps)),
        "median_estimator": f"streaming angular histogram ({HISTOGRAM_STEP_DEG:g} degree bins)",
        "methods": {},
    }
    method_report: Dict[str, object] = {}
    for method in methods:
        metrics = stats_by_method[method.name].metrics()
        metrics.update(
            {
                "available_depth_frames": len(method.frames),
                "depth_dir": str(method.depth_dir),
                "id_map": str(method.id_map_path) if method.id_map_path else None,
                "depth_scale": method.depth_scale,
            }
        )
        method_report[method.name] = {
            key: finite_or_none(value) for key, value in metrics.items()
        }
    report["methods"] = method_report

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "normal_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_text_report(args.out / "normal_metrics.txt", report)

    fieldnames = ["method", "frame_id"] + [
        key for key in per_frame_rows[0].keys() if key not in ("method", "frame_id")
    ]
    with (args.out / "normal_metrics_per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame_rows:
            writer.writerow({key: finite_or_none(value) for key, value in row.items()})

    write_cdf(args.out, stats_by_method)
    print(
        f"Evaluated {len(common_frames)} exact frames at {eval_width}x{eval_height}; "
        f"saved results to {args.out}"
    )


if __name__ == "__main__":
    main()
