"""Finite-attempt Out-Of-Memory (OOM) recovery and checkpoint preservation.

Provides:
- Limited-retry OOM recovery mechanism.
- Emergency checkpoint preservation upon memory exhaustion.
- Systematic parameter mitigation (batch size halving, gradient accumulation doubling).
- Strict audit logging to prevent silent changes to experimental configurations.
- Deterministic halt and MRE preservation if retries are exhausted.
"""
from dataclasses import asdict
import datetime
import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class OOMRecoveryManager:
    """Manages training recovery from Out-Of-Memory events with finite retries."""

    def __init__(self, max_retries: int = 3, log_path: Optional[Path] = None):
        self.max_retries = max_retries
        self.log_path = log_path
        self.attempts = 0
        self.history: list[Dict[str, Any]] = []

    def is_oom_error(self, exc: BaseException) -> bool:
        """Check if an exception is a GPU out-of-memory error."""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda out of memory" in msg or "hip out of memory" in msg:
            return True
        return False

    def handle_oom(
        self,
        exc: BaseException,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        output_dir: Path,
        current_batch_size: int,
        current_accum_steps: int = 1,
    ) -> Tuple[bool, int, int]:
        """Handle an OOM exception.
        
        Returns:
            Tuple of (can_recover: bool, new_batch_size: int, new_accum_steps: int)
        """
        self.attempts += 1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Clean memory cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Save emergency checkpoint
        emergency_path = output_dir / f"oom_emergency_epoch_{epoch:03d}_step_{step:06d}.pt"
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "oom_error": str(exc),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            torch.save(checkpoint_data, emergency_path)
            logger.info("Saved emergency OOM checkpoint to %s", emergency_path)
        except Exception as save_err:
            logger.error("Failed to save emergency checkpoint: %s", save_err)

        # 3. Check retry bounds
        if self.attempts > self.max_retries:
            logger.error(
                "OOM recovery failed: exceeded maximum retries (%d/%d). Halting.",
                self.attempts,
                self.max_retries,
            )
            self._log_event(epoch, step, current_batch_size, current_accum_steps, 0, 0, str(exc), recovered=False)
            return False, current_batch_size, current_accum_steps

        # 4. Compute parameter mitigations:
        # Halve physical batch size (floor at 16) and double accumulation steps
        new_batch_size = max(16, current_batch_size // 2)
        new_accum_steps = current_accum_steps * 2

        if new_batch_size == current_batch_size and self.attempts > 1:
            # Cannot shrink batch size further
            logger.error("OOM recovery halted: batch size is already at minimum threshold (%d).", new_batch_size)
            self._log_event(epoch, step, current_batch_size, current_accum_steps, new_batch_size, new_accum_steps, str(exc), recovered=False)
            return False, current_batch_size, current_accum_steps

        logger.warning(
            "OOM Recovery Attempt %d/%d at epoch %d, step %d: adjusting batch_size %d -> %d, accum_steps %d -> %d",
            self.attempts,
            self.max_retries,
            epoch,
            step,
            current_batch_size,
            new_batch_size,
            current_accum_steps,
            new_accum_steps,
        )

        self._log_event(
            epoch,
            step,
            current_batch_size,
            current_accum_steps,
            new_batch_size,
            new_accum_steps,
            str(exc),
            recovered=True,
            checkpoint_saved=str(emergency_path),
        )

        return True, new_batch_size, new_accum_steps

    def _log_event(
        self,
        epoch: int,
        step: int,
        old_batch: int,
        old_accum: int,
        new_batch: int,
        new_accum: int,
        error_msg: str,
        recovered: bool,
        checkpoint_saved: Optional[str] = None,
    ) -> None:
        event = {
            "attempt": self.attempts,
            "max_retries": self.max_retries,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "epoch": epoch,
            "step": step,
            "old_batch_size": old_batch,
            "old_accum_steps": old_accum,
            "new_batch_size": new_batch,
            "new_accum_steps": new_accum,
            "recovered": recovered,
            "error": error_msg,
            "emergency_checkpoint": checkpoint_saved,
        }
        self.history.append(event)
        if self.log_path:
            try:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "w") as f:
                    json.dump(self.history, f, indent=2)
            except Exception as e:
                logger.error("Failed to write OOM recovery log: %s", e)
