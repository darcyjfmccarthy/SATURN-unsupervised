"""Bounded smoke CLI, separate-process resume and early failure contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class SmokeChecks(unittest.TestCase):
    def test_two_cpu_device_smoke_and_fresh_process_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            command = [sys.executable, "scripts/smoke_saturn_jax.py", "--device", "cpu",
                       "--expected-local-device-count", "2", "--mixed-precision", "bf16", "--output-dir", str(output)]
            environment = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
            for phase, expected_step in (("initial", 1), ("resume", 2)):
                result = subprocess.run(command + ["--phase", phase], cwd=Path(__file__).resolve().parents[2],
                                        env=environment, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / f"{phase}_receipt.json").read_text())
                self.assertIs(report["success"], True)
                self.assertEqual(report["platform"], "cpu")
                self.assertEqual(report["step"], expected_step)
                self.assertEqual(report["local_device_count"], 2)
                self.assertIsNone(report["compute_chip_hours"])
                self.assertTrue(report["timestamp_utc"].endswith("+00:00"))
                with np.load(output / f"{phase}_embeddings.npz", allow_pickle=False) as artifact:
                    self.assertEqual(artifact["embeddings"].shape, (21, 8))
                    self.assertEqual(artifact["macrogenes"].shape, (21, 8))
                    self.assertEqual(artifact["embeddings"].dtype, np.float32)
            repeated = subprocess.run(command + ["--phase", "resume"], env=environment,
                                      capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("immutable", repeated.stderr)
            missing = subprocess.run([sys.executable, "scripts/smoke_saturn_jax.py", "--phase", "initial",
                "--output-dir", str(Path(directory) / "wrong_platform"), "--device", "tpu",
                "--expected-local-device-count", "1"], env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse((Path(directory) / "wrong_platform").exists())


if __name__ == "__main__":
    unittest.main()
