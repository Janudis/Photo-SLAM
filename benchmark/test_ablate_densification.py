#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate_densification as ablation  # noqa: E402
import run as benchmark_run  # noqa: E402


class DensificationAblationTest(unittest.TestCase):
    def test_every_variant_uses_the_fixed_mvs_tsdf_setup(self) -> None:
        ablation.validate_variant_matrix(ablation.VARIANTS.values())
        for variant in ablation.VARIANTS.values():
            values = variant.overrides
            self.assertEqual(
                values["Mapper.monocular_rendered_depth_densify"], 0
            )
            self.assertEqual(values["Mapper.monocular_mvs_densify"], 0)
            self.assertEqual(
                values["Mapper.monocular_mvs_tsdf_evidence"], 1
            )
            self.assertEqual(
                values["Mapper.monocular_mvs_tsdf_evidence_trunc_vox"], 2.0
            )
            self.assertGreater(values["Optimization.lambda_monocular_depth"], 0)
            self.assertTrue(variant.requires_mvs)

    def test_core_suite_contains_every_variant(self) -> None:
        self.assertEqual(set(ablation.SUITES["core"]), set(ablation.VARIANTS))

    def test_pruning_variants_are_controlled_changes(self) -> None:
        pairs = {
            "surface_views": "sdf_only",
            "surface_views_final": "surface_views",
            "mvs_consistency": "sdf_only",
            "combined": "surface_views",
            "combined_final": "combined",
        }
        for variant_key, baseline_key in pairs.items():
            variant = ablation.VARIANTS[variant_key].overrides
            baseline = ablation.VARIANTS[baseline_key].overrides
            differing = {
                key
                for key in set(variant) | set(baseline)
                if variant.get(key) != baseline.get(key)
            }
            self.assertEqual(len(differing), 1, variant_key)

    def test_all_overrides_apply_to_canonical_replica_yaml(self) -> None:
        source = (
            ablation.REPO_ROOT
            / "cfg/voxel_mapper/Monocular/Replica/replica_mono_voxel.yaml"
        ).read_text(encoding="utf-8")
        for variant in ablation.VARIANTS.values():
            result = benchmark_run.apply_yaml_overrides(
                source, variant.overrides
            )
            self.assertIn(
                f"Mapper.monocular_rendered_depth_densify: "
                f"{variant.overrides['Mapper.monocular_rendered_depth_densify']}",
                result,
            )

    def test_aggregation_and_metric_winners(self) -> None:
        variants = (
            ablation.VARIANTS["sdf_only"],
            ablation.VARIANTS["combined"],
        )

        def row(variant: str, trial: int, offset: float) -> dict[str, float | int | str]:
            return {
                "variant": variant,
                "trial": trial,
                "accuracy_cm": 10.0 - offset,
                "completeness_cm": 11.0 - offset,
                "completion_ratio_percent": 40.0 + offset,
                "precision_5cm": 0.4 + offset / 100.0,
                "recall_5cm": 0.4 + offset / 100.0,
                "fscore_5cm": 0.4 + offset / 100.0,
                "psnr": 20.0 + offset,
                "ssim": 0.7 + offset / 100.0,
                "lpips": 0.4 - offset / 100.0,
            }

        rows = [
            row("sdf_only", 1, 0.0),
            row("combined", 1, 1.0),
            row("combined", 2, 3.0),
        ]
        aggregated = ablation.aggregate_trials(variants, rows)
        self.assertEqual(aggregated["combined"]["completed_trials"], 2)
        self.assertAlmostEqual(
            aggregated["combined"]["mean"]["psnr"], 22.0
        )
        winners = ablation.best_by_metric(aggregated)
        self.assertEqual(winners["accuracy_cm"]["variant"], "combined")
        self.assertEqual(winners["psnr"]["variant"], "combined")


if __name__ == "__main__":
    unittest.main()
