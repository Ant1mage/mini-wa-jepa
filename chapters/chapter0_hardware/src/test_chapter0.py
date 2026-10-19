import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chapters.chapter0_hardware.src.calibration import recommend_training_parameters
from chapters.chapter0_hardware.src.hardware import HardwareProfile, detect_hardware
from chapters.chapter0_hardware.src.oom_recovery import OOMRecoveryManager
from chapters.chapter0_hardware.src.operators import sdpa_context, verify_attention_fidelity


class Chapter0Tests(unittest.TestCase):
    def test_hardware_profiling(self):
        profile = detect_hardware()
        self.assertIsInstance(profile, HardwareProfile)
        self.assertGreater(profile.cpu.physical_cores, 0)
        self.assertGreater(profile.cpu.total_ram_gib, 0)

    def test_parameter_calibration(self):
        profile = detect_hardware()
        profile.gpu.total_memory_gib = 20.0
        rec_20 = recommend_training_parameters(profile)
        self.assertEqual(rec_20.batch_size, 128)
        self.assertEqual(rec_20.gradient_accumulation_steps, 1)

        profile.gpu.total_memory_gib = 8.0
        rec_8 = recommend_training_parameters(profile)
        self.assertEqual(rec_8.batch_size, 32)
        self.assertEqual(rec_8.gradient_accumulation_steps, 4)

    def test_sdpa_context_and_fallback(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(2, 4, 16, 32, device=device)
        k, v = torch.randn_like(q), torch.randn_like(q)
        with sdpa_context("MATH"):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, q.shape)

        with sdpa_context("AUTO"):
            out_auto = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out_auto.shape, q.shape)

    def test_oom_recovery(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            mgr = OOMRecoveryManager(max_retries=2, log_path=out_path / "oom_log.json")
            model = torch.nn.Linear(8, 8)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            exc = RuntimeError("CUDA out of memory")

            can_recover, new_bs, new_accum = mgr.handle_oom(
                exc, model, optimizer, None, epoch=1, step=5, output_dir=out_path,
                current_batch_size=128, current_accum_steps=1
            )
            self.assertTrue(can_recover)
            self.assertEqual(new_bs, 64)
            self.assertEqual(new_accum, 2)
            self.assertTrue((out_path / "oom_emergency_epoch_001_step_000005.pt").exists())


if __name__ == "__main__":
    unittest.main()
