#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run.py")
PATCH_PATH = Path(__file__).with_name("patches") / "tandem-blackwell.patch"
SPEC = importlib.util.spec_from_file_location("baseline_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


class BaselineLauncherTest(unittest.TestCase):
    def test_replica_sequence_set_matches_paper_protocol(self):
        sequences = baseline.expand_sequences(
            "replica", ["all"], Path("/unused")
        )
        self.assertEqual(sequences, baseline.REPLICA_SEQUENCES)
        self.assertEqual(len(sequences), 8)

    def test_scannet_sequence_set_matches_requested_protocol(self):
        sequences = baseline.expand_sequences(
            "scannet", ["all"], Path("/unused")
        )
        self.assertEqual(sequences, baseline.SCANNET_SEQUENCES)
        self.assertEqual(len(sequences), 6)

    def test_monogs_config_uses_read_only_prepared_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = baseline.PreparedInput(
                root / "input",
                root / "input/results",
                root / "input/depths",
                baseline.CameraCalibration(600, 600, 599.5, 339.5, 1200, 680),
                root / "input/calib.txt",
                2000,
            )
            output = root / "output"
            output.mkdir()
            config = baseline.monogs_config(
                "replica", "office0", prepared, output
            )
            contents = config.read_text(encoding="utf-8")
            self.assertIn('sensor_type: "monocular"', contents)
            self.assertIn(str(prepared.root), contents)
            self.assertIn(str(output / "native_runs"), contents)

    def test_latest_monogs_run_uses_timestamp_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run = output / "native_runs/2026-08-28-12-00-00"
            point_cloud = run / "point_cloud/final/point_cloud.ply"
            point_cloud.parent.mkdir(parents=True)
            point_cloud.write_bytes(b"ply\n")
            self.assertEqual(baseline.latest_monogs_run(output), run)

    def test_natural_path_order_preserves_scannet_frame_order(self):
        paths = [Path("10.jpg"), Path("2.jpg"), Path("1.jpg")]
        self.assertEqual(
            [path.name for path in sorted(paths, key=baseline.natural_path_key)],
            ["1.jpg", "2.jpg", "10.jpg"],
        )

    def test_tandem_command_records_explicit_preset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = baseline.PreparedInput(
                root / "input",
                root / "input/colors",
                None,
                baseline.CameraCalibration(600, 600, 599.5, 339.5, 1200, 680),
                root / "input/calib.txt",
                2000,
            )
            prepared.root.mkdir(parents=True)
            output = root / "output"
            output.mkdir()
            command = baseline.command_for(
                "tandem",
                "replica",
                "office0",
                prepared,
                output,
                "dataset",
            )
            self.assertIn("preset=dataset", command)
            self.assertIn("mesh_extraction_freq=0", command)

    def test_tandem_patch_uses_scoped_boost_placeholders(self):
        contents = PATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("boost::placeholders::_1", contents)
        self.assertIn("boost::placeholders::_4", contents)
        self.assertNotIn("BOOST_BIND_GLOBAL_PLACEHOLDERS", contents)
        self.assertIn("-std=c++17", contents)
        self.assertIn(
            "set_property(TARGET dr-mvsnet PROPERTY CXX_STANDARD 17)",
            contents,
        )
        self.assertIn("PRIVATE HAS_PANGOLIN", contents)
        self.assertIn("#ifdef HAS_PANGOLIN", contents)
        self.assertIn("CUDA::nvToolsExt", contents)

    def test_run_lock_rejects_duplicate_active_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / ".run.lock"
            first = baseline.acquire_run_lock(lock_path)
            try:
                with self.assertRaisesRegex(
                    baseline.BaselineError, "already active"
                ):
                    baseline.acquire_run_lock(lock_path)
            finally:
                first.close()

            reopened = baseline.acquire_run_lock(lock_path)
            reopened.close()

    def test_monitor_terminates_native_process_after_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = baseline.run_monitored(
                [
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "print('Traceback (most recent call last):', flush=True); "
                        "time.sleep(30)"
                    ),
                ],
                root,
                root / "console.log",
            )
            self.assertNotEqual(metrics["return_code"], 0)
            self.assertLess(metrics["wall_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()
