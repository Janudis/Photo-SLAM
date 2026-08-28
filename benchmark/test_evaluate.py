#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as benchmark_evaluate  # noqa: E402


class EvaluateTest(unittest.TestCase):
    def test_rotation_to_quaternion_identity(self) -> None:
        actual = np.asarray(
            benchmark_evaluate.rotation_to_quaternion(np.eye(3))
        )
        np.testing.assert_allclose(actual, [0.0, 0.0, 0.0, 1.0])

    def test_rotation_to_quaternion_half_turn(self) -> None:
        rotation = np.diag([1.0, -1.0, -1.0])
        actual = np.asarray(
            benchmark_evaluate.rotation_to_quaternion(rotation)
        )
        expected = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(
            np.allclose(actual, expected) or np.allclose(actual, -expected)
        )

    def test_materialize_replica_tum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "traj.txt"
            output = root / "traj_tum.txt"
            identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
            source.write_text(identity + "\n" + identity + "\n", encoding="utf-8")

            benchmark_evaluate.materialize_replica_tum(source, output)

            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0].split()[0]), 0.0)
            self.assertAlmostEqual(float(rows[1].split()[0]), 1.0 / 30.0)
            self.assertEqual(len(rows[0].split()), 8)

    def test_parse_keyed_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "psnr.txt"
            path.write_text("# key value\n2 30.5\n7 31.25\n", encoding="utf-8")
            self.assertEqual(
                benchmark_evaluate.parse_keyed_metric(path),
                {2: 30.5, 7: 31.25},
            )


if __name__ == "__main__":
    unittest.main()
