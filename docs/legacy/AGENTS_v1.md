# Project working agreement

- Read GOAL.md for the user's additional design requirements. Retain the full
  baseline/ablation list from the chat; the shorter file list does not cancel it.
- The public Python package name is mini_wa_jepa. Stage 2 must introduce the
  replaceable BEVProvider contract (CHW uint8 + metadata), simulator-ground-truth
  implementation and configurable BEV noise; models must not depend on simulator
  truth-specific fields. Keep later noisy-BEV robustness comparisons in scope.

- Work only on the stage explicitly approved by the user. Stages 1 and 2 are
  authorized; the user confirmed stage 2 after reviewing GOAL.md. Stage 2 pauses after 5k frames before
  expansion to 100k+. Report changes, numeric acceptance results, problems and
  proposed next step at each gate, then stop.
- Maintain the current stage checklist in TASKS.md and give progress updates.
- Use MetaDrive; inspect documentation/source or run a small experiment before
  assuming uncertain simulator APIs.
- PyTorch ROCm only. No CUDA runtime dependencies, flash-attn, xformers, Triton,
  custom CUDA operators or torch.compile. Attention uses functional SDPA; start
  with the MATH backend verified in scripts/check_rocm.py.
- No WA-JEPA source, clones, weights or datasets; no V-JEPA weights. Implement
  model code independently from the architecture described in the user brief.
- Keep the training stack small: PyTorch, numpy, dataclass config, CSV/TensorBoard.
  No RGB model input, perception models, F1, or custom simulator.
- Record actual results, failures, config, seeds and code commit. Never label an
  episode collision or timeout as a program crash or route success.
- Tests should include meaningful minimal shape/numeric or end-to-end smoke
  checks. Do not mark later training or baselines done before they are run.
- User added live desktop requirements: expert and dataset replay support --live;
  composite camera/BEV/waypoints/status; Space/N/R/Q; M toggles WASD takeover.
  scripts/render_episode.py saves MP4 under outputs/videos/<stage>/ and supports
  --headless. Every stage must include an acceptance video. Stage 3 onward also
  requires TensorBoard plus periodic prediction-vs-ground-truth images.
