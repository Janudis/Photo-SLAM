#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_tum as tum  # noqa: E402


class TumEvaluateTest(unittest.TestCase):
    def test_load_voxel_jobs_requires_complete_two_method_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for sequence in tum.TUM_SEQUENCES:
                for method in tum.VOXEL_METHODS:
                    shutdown = (
                        root / "tum" / sequence / method / "trial_01" / "10_shutdown"
                    )
                    shutdown.mkdir(parents=True)
                    jobs.append(
                        {
                            "status": "complete",
                            "method": method,
                            "sequence": sequence,
                            "trial": 1,
                            "shutdown_dir": str(shutdown),
                        }
                    )
            (root / "run_manifest.json").write_text(
                json.dumps({"status": "complete", "dataset": "tum", "jobs": jobs}),
                encoding="utf-8",
            )

            loaded = tum.load_voxel_jobs(root)

            self.assertEqual(len(loaded), 6)

    def test_load_hislam2_rows_uses_after_opt_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for index, sequence in enumerate(tum.TUM_SEQUENCES):
                metrics = (
                    root
                    / "tum"
                    / sequence
                    / "hislam2"
                    / "trial_01"
                    / "native"
                    / "psnr"
                    / "after_opt"
                    / "final_result.json"
                )
                metrics.parent.mkdir(parents=True)
                metrics.write_text(
                    json.dumps(
                        {
                            "mean_psnr": 20 + index,
                            "mean_ssim": 0.7 + index / 100,
                            "mean_lpips": 0.3 - index / 100,
                        }
                    ),
                    encoding="utf-8",
                )
                render_dir = metrics.parents[2] / "renders" / "image_after_opt"
                render_dir.mkdir(parents=True)
                for frame in range(index + 1):
                    (render_dir / f"{frame:06d}.jpg").write_bytes(b"image")
                jobs.append(
                    {
                        "status": "complete",
                        "sequence": sequence,
                        "trial": 1,
                        "frame_count": 100 + index,
                        "appearance_metrics": str(metrics),
                    }
                )
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "dataset": "tum",
                        "method": "hislam2",
                        "jobs": jobs,
                    }
                ),
                encoding="utf-8",
            )

            rows = tum.load_hislam2_rows(root)

            self.assertEqual(rows[tum.TUM_SEQUENCES[1]]["psnr"], 21.0)
            self.assertEqual(rows[tum.TUM_SEQUENCES[2]]["evaluated_frames"], 3)

    def test_latex_rows_use_table_order(self):
        per_sequence = {
            sequence: {
                method: {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2}
                for method in ("hislam2", *tum.VOXEL_METHODS)
            }
            for sequence in tum.TUM_SEQUENCES
        }
        summary = {
            "sequences": per_sequence,
            "average": {
                method: {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2}
                for method in ("hislam2", *tum.VOXEL_METHODS)
            },
        }

        latex = tum.latex_rows(summary)

        self.assertIn(r"\multirow{3}{*}{HI-SLAM2}", latex)
        self.assertIn(r"& PSNR $\uparrow$", latex)
        self.assertIn(r"\multirow{3}{*}{Ours+MVS}", latex)


if __name__ == "__main__":
    unittest.main()
