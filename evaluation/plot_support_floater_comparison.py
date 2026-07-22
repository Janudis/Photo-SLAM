#!/usr/bin/env python3

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def read_curve(path: Path):
    thresholds = []
    counts = []
    fractions = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "distance_threshold_cm" not in reader.fieldnames:
            raise ValueError(f"invalid floater CSV: {path}")
        count_column = next(
            (name for name in reader.fieldnames if name.endswith("_farther") and name != "fraction_farther"),
            None,
        )
        if count_column is None or "fraction_farther" not in reader.fieldnames:
            raise ValueError(f"missing count/fraction columns: {path}")
        for row in reader:
            thresholds.append(float(row["distance_threshold_cm"]))
            counts.append(int(row[count_column]))
            fractions.append(float(row["fraction_farther"]))
    if not thresholds:
        raise ValueError(f"empty floater CSV: {path}")
    return np.asarray(thresholds), np.asarray(counts), np.asarray(fractions)


def infer_total(counts, fractions):
    estimates = [
        int(round(count / fraction))
        for count, fraction in zip(counts, fractions)
        if count > 0 and fraction > 0.0
    ]
    if not estimates:
        return 0
    return Counter(estimates).most_common(1)[0][0]


def draw_text(image, text, origin, scale=0.6, color=(55, 55, 55), thickness=1):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def nice_axis_step(value):
    if value <= 0:
        return 1
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1, 2, 5, 10):
        if normalized <= candidate:
            return int(candidate * magnitude)
    return int(10 * magnitude)


def main():
    parser = argparse.ArgumentParser(description="Plot voxel/Gaussian primitive floater curves")
    parser.add_argument("--voxel-csv", required=True, type=Path)
    parser.add_argument("--gaussian-csv", required=True, type=Path)
    parser.add_argument(
        "--hislam2-csv",
        type=Path,
        help="optional HI-SLAM2 Gaussian support floater CSV",
    )
    parser.add_argument(
        "--hislam2-label",
        default="HI-SLAM2 3-sigma Gaussians",
        help="legend label for the optional HI-SLAM2 curve",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    vx, vc, vf = read_curve(args.voxel_csv)
    gx, gc, gf = read_curve(args.gaussian_csv)
    hislam2_curve = read_curve(args.hislam2_csv) if args.hislam2_csv else None
    voxel_total = infer_total(vc, vf)
    gaussian_total = infer_total(gc, gf)
    hislam2_total = (
        infer_total(hislam2_curve[1], hislam2_curve[2])
        if hislam2_curve is not None
        else 0
    )

    min_x = 5.0
    voxel_keep = vx >= min_x
    gaussian_keep = gx >= min_x
    vx, vc = vx[voxel_keep], vc[voxel_keep]
    gx, gc = gx[gaussian_keep], gc[gaussian_keep]
    if vx.size == 0 or gx.size == 0:
        raise ValueError("floater curves do not contain the 5 cm evaluation range")
    curves = [(vx, vc), (gx, gc)]
    if hislam2_curve is not None:
        hx, hc, _ = hislam2_curve
        hislam2_keep = hx >= min_x
        hx, hc = hx[hislam2_keep], hc[hislam2_keep]
        if hx.size == 0:
            raise ValueError("HI-SLAM2 floater curve does not contain the 5 cm evaluation range")
        curves.append((hx, hc))
    max_x = float(max(xs.max() for xs, _ in curves))

    curve_max = int(max(counts.max() for _, counts in curves))
    y_step = nice_axis_step(curve_max / 9.0)
    y_max = int(math.ceil(curve_max / y_step) * y_step)

    width = 1400
    height = 900 if hislam2_curve is not None else 850
    left, right = 115, width - 55
    top = 180 if hislam2_curve is not None else 145
    bottom = height - 155
    image = np.full((height, width, 3), 250, dtype=np.uint8)

    grid = (224, 224, 224)
    axis = (70, 70, 70)
    voxel_color = (125, 125, 25)
    gaussian_color = (55, 95, 215)
    hislam2_color = (190, 105, 35)

    for value in range(0, y_max + 1, y_step):
        y = int(round(bottom - (bottom - top) * value / y_max))
        cv2.line(image, (left, y), (right, y), grid, 1, cv2.LINE_AA)
        draw_text(image, f"{value:,}", (30, y + 6), 0.5)

    x_tick = 5
    tick = int(min_x)
    while tick <= int(np.ceil(max_x)):
        x = int(round(left + (right - left) * (tick - min_x) / (max_x - min_x)))
        cv2.line(image, (x, top), (x, bottom), grid, 1, cv2.LINE_AA)
        draw_text(image, str(tick), (x - 9, bottom + 28), 0.5)
        tick += x_tick

    cv2.line(image, (left, bottom), (right, bottom), axis, 2, cv2.LINE_AA)
    cv2.line(image, (left, top), (left, bottom), axis, 2, cv2.LINE_AA)

    def cropped_curve_points(xs, counts):
        return np.asarray(
            [
                [
                    int(round(left + (right - left) * (x - min_x) / (max_x - min_x))),
                    int(round(bottom - (bottom - top) * float(count) / y_max)),
                ]
                for x, count in zip(xs, counts)
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

    cv2.polylines(image, [cropped_curve_points(vx, vc)], False, voxel_color, 4, cv2.LINE_AA)
    cv2.polylines(image, [cropped_curve_points(gx, gc)], False, gaussian_color, 4, cv2.LINE_AA)
    if hislam2_curve is not None:
        cv2.polylines(
            image,
            [cropped_curve_points(hx, hc)],
            False,
            hislam2_color,
            4,
            cv2.LINE_AA,
        )

    draw_text(image, "Primitive Support Floater Comparison", (left, 48), 0.9, (35, 35, 35), 2)
    draw_text(
        image,
        "Cumulative primitives beyond the 5 cm floater threshold",
        (left, 82),
        0.58,
        (75, 75, 75),
        1,
    )
    draw_text(image, "Distance from GT surface (cm)", ((left + right) // 2 - 125, bottom + 68), 0.58)
    draw_text(image, "Primitives farther", (12, top - 14), 0.52)

    legend_y = 112
    cv2.line(image, (left, legend_y), (left + 45, legend_y), voxel_color, 5, cv2.LINE_AA)
    draw_text(image, "SVRecon zero-crossing voxels", (left + 58, legend_y + 6), 0.55)
    legend_x = left + 410
    cv2.line(image, (legend_x, legend_y), (legend_x + 45, legend_y), gaussian_color, 5, cv2.LINE_AA)
    draw_text(image, "Original Photo-SLAM 3-sigma Gaussians", (legend_x + 58, legend_y + 6), 0.55)
    if hislam2_curve is not None:
        hislam2_legend_y = 148
        cv2.line(
            image,
            (left, hislam2_legend_y),
            (left + 45, hislam2_legend_y),
            hislam2_color,
            5,
            cv2.LINE_AA,
        )
        draw_text(
            image,
            args.hislam2_label,
            (left + 58, hislam2_legend_y + 6),
            0.55,
        )

    totals_footer = (
        f"Final scene totals: {voxel_total:,} zero-crossing voxels  |  "
        f"{gaussian_total:,} Photo-SLAM Gaussians"
    )
    if hislam2_curve is not None:
        totals_footer += f"  |  {hislam2_total:,} HI-SLAM2 Gaussians"
    draw_text(image, totals_footer, (left, height - 62), 0.54, (55, 55, 55), 1)
    draw_text(
        image,
        "Finite support: voxel boundary / Gaussian 3-sigma ellipsoid; 32 probes per primitive",
        (left, height - 34),
        0.52,
        (70, 70, 70),
        1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), image):
        raise RuntimeError(f"failed to write: {args.out}")
    print(args.out)


if __name__ == "__main__":
    main()
