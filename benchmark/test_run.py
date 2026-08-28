#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run as benchmark


class BenchmarkLauncherTest(unittest.TestCase):
    def test_yaml_overrides_replace_only_requested_values(self) -> None:
        source = (
            "%YAML:1.0\n"
            "Mapper.monocular_rendered_depth_densify: 0\n"
            "Mapper.monocular_mvs_densify: 1\n"
        )
        result = benchmark.apply_yaml_overrides(
            source,
            {
                "Mapper.monocular_rendered_depth_densify": 1,
                "Mapper.monocular_mvs_densify": 0,
            },
        )
        self.assertIn("Mapper.monocular_rendered_depth_densify: 1", result)
        self.assertIn("Mapper.monocular_mvs_densify: 0", result)

    def test_yaml_override_rejects_missing_key(self) -> None:
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.apply_yaml_overrides("%YAML:1.0\n", {"Missing": 1})

    def test_yaml_override_can_append_evaluation_output_key(self) -> None:
        result = benchmark.apply_yaml_overrides(
            "%YAML:1.0\nRecord.record_rendered_image: 1\n",
            {"Record.save_rendered_mesh_eval": 1},
            append_missing=True,
        )
        self.assertEqual(result.count("Record.save_rendered_mesh_eval"), 1)
        self.assertTrue(result.endswith("Record.save_rendered_mesh_eval: 1\n"))

    def test_data_resolution_accepts_replica_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "Replica/office0"
            (sequence / "results").mkdir(parents=True)
            (sequence / "traj.txt").touch()
            actual = benchmark.resolve_data_dir(
                root, benchmark.DATASETS["replica"], "office0"
            )
            self.assertEqual(actual, sequence)

    def test_all_method_expansion_is_stable(self) -> None:
        self.assertEqual(
            benchmark.expand_methods(["all"]), benchmark.DEFAULT_METHODS
        )

    def test_final_mvs_tsdf_presets_are_explicitly_selectable(self) -> None:
        selected = benchmark.expand_methods(
            ["ours_mvs_tsdf_geometry", "ours_mvs_tsdf_rendering"]
        )
        self.assertEqual(
            selected,
            ("ours_mvs_tsdf_geometry", "ours_mvs_tsdf_rendering"),
        )

    def test_voxel_presets_differ_only_in_densification_selection(self) -> None:
        ours = benchmark.METHODS["ours"].voxel_overrides
        ours_mvs = benchmark.METHODS["ours_mvs"].voxel_overrides
        differing = {
            key
            for key in set(ours) | set(ours_mvs)
            if ours.get(key) != ours_mvs.get(key)
        }
        self.assertEqual(
            differing,
            {
                "Mapper.monocular_rendered_depth_densify",
                "Mapper.monocular_mvs_densify",
            },
        )

    def test_final_mvs_tsdf_presets_differ_only_in_mvs_pruning(self) -> None:
        geometry = benchmark.METHODS[
            "ours_mvs_tsdf_geometry"
        ].voxel_overrides
        rendering = benchmark.METHODS[
            "ours_mvs_tsdf_rendering"
        ].voxel_overrides
        differing = {
            key
            for key in set(geometry) | set(rendering)
            if geometry.get(key) != rendering.get(key)
        }
        self.assertEqual(
            differing, {"Optimization.prune_mvs_consistency_enable"}
        )
        self.assertEqual(
            geometry["Optimization.prune_mvs_consistency_enable"], 1
        )
        self.assertEqual(
            rendering["Optimization.prune_mvs_consistency_enable"], 0
        )
        for key, value in benchmark.MVS_TSDF_COMMON_OVERRIDES.items():
            if key != "Optimization.prune_mvs_consistency_enable":
                self.assertEqual(geometry[key], value)
            self.assertEqual(rendering[key], value)

    def test_final_mvs_tsdf_presets_cover_all_replica_scene_extents(self) -> None:
        self.assertEqual(
            benchmark.MVS_TSDF_COMMON_OVERRIDES[
                "Model.global_scene_extent"
            ],
            256.0,
        )


if __name__ == "__main__":
    unittest.main()
