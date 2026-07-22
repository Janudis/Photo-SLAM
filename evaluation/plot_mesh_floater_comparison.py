#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from plot_support_floater_comparison import draw_text, nice_axis_step, read_curve


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative mesh-surface floater curves"
    )
    parser.add_argument("--ours-csv", required=True, type=Path)
    parser.add_argument("--photoslam-csv", required=True, type=Path)
    parser.add_argument("--hislam2-csv", required=True, type=Path)
    parser.add_argument("--nvblox-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    series = [
        ("Ours (SVRecon)", args.ours_csv, (190, 75, 225)),
        ("Original Photo-SLAM", args.photoslam_csv, (55, 95, 215)),
        ("HI-SLAM2 after refinement", args.hislam2_csv, (190, 105, 35)),
        ("nvblox online", args.nvblox_csv, (75, 155, 80)),
    ]

    min_x = 5.0
    curves = []
    totals = []
    for label, path, color in series:
        thresholds, counts, fractions = read_curve(path)
        keep = thresholds >= min_x
        thresholds = thresholds[keep]
        counts = counts[keep]
        if thresholds.size == 0:
            raise ValueError(f"{path} does not contain the 5 cm evaluation range")
        total_candidates = [
            int(round(count / fraction))
            for count, fraction in zip(counts, fractions[keep])
            if count > 0 and fraction > 0.0
        ]
        totals.append(total_candidates[0] if total_candidates else 0)
        curves.append((label, thresholds, counts, color))

    max_x = float(max(thresholds.max() for _, thresholds, _, _ in curves))
    curve_max = int(max(counts.max() for _, _, counts, _ in curves))
    y_step = nice_axis_step(curve_max / 9.0)
    y_max = max(y_step, int(math.ceil(curve_max / y_step) * y_step))

    width, height = 1400, 900
    left, right = 115, width - 55
    top, bottom = 180, height - 155
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    grid = (224, 224, 224)
    axis = (70, 70, 70)

    for value in range(0, y_max + 1, y_step):
        y = int(round(bottom - (bottom - top) * value / y_max))
        cv2.line(image, (left, y), (right, y), grid, 1, cv2.LINE_AA)
        draw_text(image, f"{value:,}", (30, y + 6), 0.5)

    for tick in range(int(min_x), int(math.ceil(max_x)) + 1, 5):
        x = int(round(left + (right - left) * (tick - min_x) / (max_x - min_x)))
        cv2.line(image, (x, top), (x, bottom), grid, 1, cv2.LINE_AA)
        draw_text(image, str(tick), (x - 9, bottom + 28), 0.5)

    cv2.line(image, (left, bottom), (right, bottom), axis, 2, cv2.LINE_AA)
    cv2.line(image, (left, top), (left, bottom), axis, 2, cv2.LINE_AA)

    for _, thresholds, counts, color in curves:
        points = np.asarray(
            [
                [
                    int(round(left + (right - left) * (x - min_x) / (max_x - min_x))),
                    int(round(bottom - (bottom - top) * float(count) / y_max)),
                ]
                for x, count in zip(thresholds, counts)
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, color, 4, cv2.LINE_AA)

    draw_text(image, "Mesh Surface Floater Comparison", (left, 48), 0.9, (35, 35, 35), 2)
    draw_text(
        image,
        "Cumulative sampled surface points beyond the 5 cm floater threshold",
        (left, 82),
        0.58,
        (75, 75, 75),
    )
    draw_text(image, "Distance from GT surface (cm)", ((left + right) // 2 - 125, bottom + 68), 0.58)
    draw_text(image, "Surface samples farther", (12, top - 14), 0.52)

    legend_positions = [(left, 112), (left + 470, 112), (left, 148), (left + 470, 148)]
    for (label, _, _, color), (x, y) in zip(curves, legend_positions):
        cv2.line(image, (x, y), (x + 45, y), color, 5, cv2.LINE_AA)
        draw_text(image, label, (x + 58, y + 6), 0.55)

    totals_text = "Evaluation samples: " + " | ".join(
        f"{label}: {total:,}" for (label, _, _), total in zip(series, totals)
    )
    draw_text(image, totals_text, (left, height - 55), 0.5, (55, 55, 55), 1)
    draw_text(
        image,
        "Each curve measures reconstructed mesh samples against the same Replica GT mesh.",
        (left, height - 28),
        0.52,
        (70, 70, 70),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), image):
        raise RuntimeError(f"failed to write: {args.out}")
    print(args.out)


if __name__ == "__main__":
    main()
