"""Chapter 5 Source Package: Perception degradation, frame dropout, and benchmark stress testing."""
from chapters.chapter5_stress_testing.src.degradation import (
    apply_boundary_loss,
    apply_front_blindspot,
    apply_random_box_occlusion,
    apply_temporal_frame_dropout,
)
from chapters.chapter5_stress_testing.src.benchmark import (
    evaluate_under_degradation,
    main,
)

__all__ = [
    "apply_front_blindspot",
    "apply_random_box_occlusion",
    "apply_boundary_loss",
    "apply_temporal_frame_dropout",
    "evaluate_under_degradation",
    "main",
]
