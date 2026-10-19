"""Unit tests for hardware auto-detection, operator selection, calibration, and OOM recovery."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from configs.stage4 import Stage4Config
from mini_wa_jepa.calibration import CalibrationRecommendation, recommend_training_parameters
from mini_wa_jepa.hardware import HardwareProfile, detect_hardware
from mini_wa_jepa.oom_recovery import OOMRecoveryManager
from mini_wa_jepa.operators import sdpa_context, verify_attention_fidelity


class HardwareDetectionTests(unittest.TestCase):
    def test_detection_runs_non_intrusively(self):
        profile = detect_hardware()
        self.assertIsInstance(profile, HardwareProfile)
        self.assertGreater(profile.cpu.physical_cores, 0)
        self.assertGreater(profile.cpu.total_ram_gib, 0)
        if torch.cuda.is_available():
            self.assertTrue(profile.gpu.available)
            self.assertGreater(profile.gpu.total_memory_gib, 0)
            self.assertIn(profile.gpu.vendor, ["AMD", "NVIDIA", "Apple", "Generic"])
        summary = profile.summary()
        self.assertIn("Hardware Environment Profile", summary)

    def test_calibration_rules_by_vram_tiers(self):
        profile = detect_hardware()
        # High VRAM test (20GB)
        profile.gpu.total_memory_gib = 20.0
        rec_20 = recommend_training_parameters(profile)
        self.assertEqual(rec_20.batch_size, 128)
        self.assertEqual(rec_20.gradient_accumulation_steps, 1)

        # Mid VRAM test (12GB)
        profile.gpu.total_memory_gib = 12.0
        rec_12 = recommend_training_parameters(profile)
        self.assertEqual(rec_12.batch_size, 64)
        self.assertEqual(rec_12.gradient_accumulation_steps, 2)
        self.assertIn("VICReg", rec_12.vicreg_batch_note)

        # Low VRAM test (8GB)
        profile.gpu.total_memory_gib = 8.0
        rec_8 = recommend_training_parameters(profile)
        self.assertEqual(rec_8.batch_size, 32)
        self.assertEqual(rec_8.gradient_accumulation_steps, 4)
        self.assertIn("CAUTION", rec_8.vicreg_batch_note)


class OperatorSelectionTests(unittest.TestCase):
    def test_sdpa_context_math_baseline(self):
        q = torch.randn(2, 4, 16, 32, device="cuda" if torch.cuda.is_available() else "cpu")
        k, v = torch.randn_like(q), torch.randn_like(q)
        with sdpa_context("MATH"):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, q.shape)

    def test_sdpa_context_auto_and_fallback(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(2, 4, 16, 32, device=device)
        k, v = torch.randn_like(q), torch.randn_like(q)
        with sdpa_context("AUTO"):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, q.shape)

        # Invalid backend should trigger graceful fallback to MATH without throwing
        with sdpa_context("NON_EXISTENT_BACKEND", allow_fallback=True):
            out_fallback = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.assertEqual(out_fallback.shape, q.shape)


class OOMRecoveryTests(unittest.TestCase):
    def test_oom_recovery_manager_workflow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            mgr = OOMRecoveryManager(max_retries=2, log_path=out_path / "oom_log.json")

            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Simulated OOM
            exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            self.assertTrue(mgr.is_oom_error(exc))

            # Attempt 1
            can_recover, new_bs, new_accum = mgr.handle_oom(
                exc, model, optimizer, None, epoch=1, step=10, output_dir=out_path,
                current_batch_size=128, current_accum_steps=1
            )
            self.assertTrue(can_recover)
            self.assertEqual(new_bs, 64)
            self.assertEqual(new_accum, 2)
            self.assertTrue((out_path / "oom_emergency_epoch_001_step_000010.pt").exists())

            # Attempt 2
            can_recover2, new_bs2, new_accum2 = mgr.handle_oom(
                exc, model, optimizer, None, epoch=1, step=11, output_dir=out_path,
                current_batch_size=64, current_accum_steps=2
            )
            self.assertTrue(can_recover2)
            self.assertEqual(new_bs2, 32)
            self.assertEqual(new_accum2, 4)

            # Attempt 3 (exceeds max_retries = 2)
            can_recover3, _, _ = mgr.handle_oom(
                exc, model, optimizer, None, epoch=1, step=12, output_dir=out_path,
                current_batch_size=32, current_accum_steps=4
            )
            self.assertFalse(can_recover3)

            # Audit log check
            log_data = json.loads((out_path / "oom_log.json").read_text())
            self.assertEqual(len(log_data), 3)
            self.assertFalse(log_data[-1]["recovered"])


if __name__ == "__main__":
    unittest.main()
