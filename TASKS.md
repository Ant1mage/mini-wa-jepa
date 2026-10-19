# Restarted plan — GOAL.md v2

## Rebuildable cache cleanup (2026-10-19)

- [x] Remove `outputs/data/stage2_v2_100k/baseline_cache` (7,889,485,824 allocated bytes) and 29 project `__pycache__` directories (1,990,656 allocated bytes), as explicitly requested
- [x] Validate all 240 original schema-2 shards (120,000 frames) with ZIP CRC checks before deletion; verify no process maps the BEV cache and no tracked files are removed
- [x] Verify all 6,494 remaining artifact/output files retain their sizes and modification times; preserve `.venv`, `.deps`, Stage 6 features, original datasets, checkpoints and recordings

Measured filesystem free-space increase: 7,891,611,648 bytes (7.35 GiB). Audit: `artifacts/cache_cleanup_20261019/cleanup.json`, including source commit and GOAL hash. Production datasets automatically rebuild the removed cache from original compressed shards; standalone chapter training needs that cache rebuilt first. No model or application source changed.

## Git history reset (2026-10-19)

User authorized removing all previous local/remote Git history and publishing the current workspace to remote `main`. This supersedes earlier Git-history retention requirements; existing source snapshots, experiment reports, failed-run evidence, datasets and model files on disk remain preserved.

- [x] Publish the current source, chapters and reports as one parentless commit on `main`; keep existing `.gitignore` exclusions and tracked Stage 1 acceptance samples
- [x] Replace remote branch tips that retain the old history, remove local historical refs/reflogs/temporary objects and verify the resulting repository

Verification: 75 tests passed; 998 snapshot files reviewed with no files over 50 MiB or credential-pattern matches. Remote `main` and existing default branch `master` both point to the same new root, and local `main` tracks `origin/main`. Old local branch/Codex snapshot refs, reflogs and Git garbage were removed; `.git` shrank from about 3,152 MiB to 89 MiB (about 2.99 GiB reclaimed). All 4,552 local data/checkpoint/recording/archive files retained their original sizes and modification times. GitHub controls eventual server-side collection of unreferenced objects; rewriting refs does not prove immediate server-side physical erasure.

## Current teaching task — Chapter0–5 first, then Chapter6 (2026-10-19)

User: “那就先把0-5改造后做chapter6”. Native marimo widgets plus existing NumPy/Matplotlib only; no additional UI framework or new videos. Preserve all formal Stage1–6 results and prior chapter implementations. Stage7 remains out of scope.

- [x] Preserve previous entry/chapters/docs in `artifacts/learning_interactions_20261019/before.tar.gz`
- [x] Refactor Chapter0–5 into three focused experiments each, with nearby controls, linked explanations, Chinese annotated code and folded formal reports
- [x] Verify Chapter0–5: all18 sections and36 visible-control operations passed actual Chromium events; chapter subsection state retained, no page errors; math regression7/7 passed and notebook syntax checks passed. Independent notebook executions also recorded under `artifacts/learning_interactions_20261019`.
- [x] Add Chapter6 using real frozen Stage6 candidate commands/actual outcomes and honest latent-prediction evidence; 32 test contexts, four candidates, 0–2s actual replay, conditioned/history-only and wrong-command comparisons
- [x] Integrate all seven chapters into the ordinary learning entry and standalone notebooks; all21 sections/45 visible-control operations passed Chromium; additional Chapter6 semantic browser checks passed; all7 standalone notebooks and unified entry passed `app.run()`; full75 tests passed in13.186s
- [x] Document final behavior and deliver the working local marimo entry (`http://127.0.0.1:2718`); evidence and source manifest under `artifacts/learning_interactions_20261019/`, current guide `docs/LEARNING_WORKBENCH.md`

The user accepted stage 4 and authorized stage 5 with “那你继续吧，先把14gb这个干掉”, as recorded in AGENTS.md, and explicitly reconfirmed starting stage 5 on 2026-10-18. Preserve existing code, reports and failed runs; schema-1 training datasets were removed under the explicit cleanup instruction. Previous progress is retained in docs/legacy/TASKS_v1.md. Stage 5 formal evaluation and delivery audit are complete; all 6 protocol/delivery gates passed. On 2026-10-19 the user explicitly authorized Stage 6: “先进行阶段6，这个再阶段6后面再做”. Chapter interaction redesign is deferred until after Stage 6; no additional frameworks.

## Stage 6: action-conditioned world prediction (completed; report delivered, awaiting user acceptance)

- [x] Record authorization and defer chapter interaction redesign until after Stage 6
- [x] Freeze protocol after verifying deterministic same-prefix branch collection; retain all smoke/failure evidence (`fresh_reset_probe_001`: eight long-prefix SC/CC contexts, all state/image differences zero; keep-repeat futures identical). Formal strict bounds restored, native fresh-object reset enabled.
- [x] Collect split-isolated real candidate commands, executed controls and aligned actual future BEVs/states (192 contexts, 768 branches plus 48 keep-repeat controls). Three complete partitions merged into `artifacts/restart_stage6/branches`; independent raw identity/command/time/hash audit passed.
- [x] Implement action-conditioned predictor preserving Stage 4, with a matched unconditional control and fixed target encoder (22,535,784 parameters; 11,675,776 trainable)
- [x] Verify causal input separation, split isolation, replay identity, command tracking and gradient paths; calibrate ROCm training (32/64 trials, fixed batch64, actual training allocator peak 0.886GiB)
- [x] Train matched runs with TensorBoard and actual prediction/target NPZ/PNG/RRD evidence (two seeds × conditioned/unconditional ×40 epochs; 72 periodic RRD, 72 PNG and 72 NPZ checked)
- [x] Evaluate same-context candidate consequences, wrong-action control, confidence intervals and limitations (all128 test branches, 32 contexts, eight seed clusters; both seeds passed all scientific gates)
- [x] Deliver report/acceptance and stop before Stage 7; chapter interaction work remains next in the queue (`artifacts/restart_stage6/REPORT.md`, acceptance.json, summary.json, dashboard.html; delivery/scientific gates both passed; 71 tests passed)

Final Stage 6: matched latent MSE 0.019036 vs0.034644 and 0.018891 vs0.034686, gains45.05%/45.54%; paired seed-cluster95% delta intervals strictly below zero. Consequence-delta error improves66.43%/66.62%; candidate-outcome retrieval78.125%/77.344% vs25% unconditional. Wrong-command MSE rises to0.06258; observed keep-repeat latent noise is0. No new trajectory-head ADE/FDE or autonomous closed-loop success rate is claimed. Test coverage is eight independent seeds, not a general driving guarantee. All earlier replay failures and source snapshots retained. Chapter interaction redesign is next per the user, restricted to existing marimo components and no added frameworks; Stage 7 is not started.

Historical preflight evidence: 71 tests passed and a two-epoch pipeline smoke completed (not formal training). Strict replay checks stopped runs001/002 and the first map partitions; retained MREs expose native object-reuse drift in traffic state. Run003 was intentionally interrupted to partition collection. Inspecting native source led to a fresh-object-reset probe (`force_destroy=True`) with unchanged previous-stage defaults. The completed formal run above supersedes this preliminary status.

Current status: stage 4 is completed and passed. Delivered deterministic 4-history/4-future world/action model and action-only fine-tuning control (both 12 epochs, selected epoch 12). Matched cohort test evaluation (9,600 frames, 22 episodes): Joint ADE 0.130488 m / FDE 0.223658 m; Action-only ADE 0.130400 m / FDE 0.225772 m; ResNet ADE 0.179834 m / FDE 0.273087 m (both beat ResNet). 20-episode held-out closed loop: Joint route completion 0.969774 (95% success, 5% collision, 0% off-road, pipeline P95 14.396 ms); Action-only route completion 0.297146 (25% success, extensive stalling from representation rank decay to 0.7435 vs Joint 0.8530). Gates: all 7 passed; per user instruction the arbitrary 14GB VRAM peak gate was removed in light of 20GB physical VRAM and mmap dataset loading (allocator peak was 10.1779 GiB, observed DRM was 14.2830 GB, well within 20GB). Deliverables (REPORT.md, acceptance.json, stage4_acceptance.mp4, dashboard.html, results_manifest.json, delivery_checks.json) completed and audited with overall passed: true. Stage 4 was accepted and Stage 5 was authorized; the current formal evaluation status is recorded below.

Historical Rerun review: opened existing expert recordings for SCS/309000 and CSC/319000 alongside a 20-episode measured overview (`artifacts/rerun_review`). Only those two episodes have full recorded frames; the other 18 have episode metrics. At that review the baseline was incomplete at 15/20 episodes; the current full evaluation supersedes that partial score without deleting it.

## Stage 1: revised environment and BEV acceptance

- [x] Compare specifications; preserve old work and failure records
- [x] Verify native SemanticCamera orthographic BEV via EGL (128x128, 64m square)
- [x] Define/test four channels, ego exclusion, visibility semantics and calibration
- [x] Recheck ROCm matmul/SDPA and forbidden dependencies
- [x] Run renderer smoke test; report API/performance mismatches
- [x] Run 100 episodes in one engine; record per-episode RAM/VRAM
- [x] Save 20 raw four-channel samples, channel previews and acceptance video
- [x] Report complete-pipeline FPS, memory trends, interfaces, limitations and gate status
- [x] Obtain stage 1 confirmation under the latest restarted agreement before any further stage 2 work (recorded in current GOAL.md §6 and AGENTS.md)

Average active pipeline: 149.81 FPS; all-inclusive wall throughput: 60.07 FPS.
Minimum per-episode average: 107.78 FPS. The stronger interpretation of a
120 FPS per-episode or reset-inclusive floor is not met. VRAM warmed range:
904.29–912.29 MiB; finite-run measurements show no sustained process VRAM growth.
The legacy same-process desktop viewer remains a known unresolved failure.

## Stage 2: v2 expert dataset and behavior-cloning baseline

- [x] Standardize on the public `RerunLogger` API; independent viewer / `.rrd` output
- [x] Add `--rerun` / `--rerun-save` to collection and replay; route `--live` to Rerun
- [x] Align BEV, ego state, telemetry and delayed realized rescue trajectories on the timeline
- [x] Remove old schema-1 training datasets per latest user instruction; retain cleanup inventory, old source and failure evidence
- [x] Add Marimo measured-report loading, ablation/ADE/FDE/open/closed-loop tables and comparisons
- [x] Document stage 3–5 periodic `.rrd` + TensorBoard + figures/video requirements
- [x] Verify SDK file output, legacy 80-frame collection, 155-frame noisy-episode replay/video and Marimo export (see artifacts/visualization_standard/run_001)
- [x] Pass 19 complete tests / 7 SDK 0.39 compatibility tests; select bundled viewer and matching protocol; verify headless stream receipt and preserve unsuccessful desktop screenshot probes
- [x] Migrate the collector to the accepted EGL 4×128×128 BEV, 10×4 `(x,y,v,heading)` labels at 0.2s, revised noise and recovery-loss semantics
- [x] Run 480-frame v2 smoke across 8 maps with third-person RGB/Rerun; verify coordinate/time/speed alignment and recovery masks
- [x] Collect/validate 100k training + 10k validation + 10k test frames; retain all noise/recovery supervision
- [x] Train 20,116,648-parameter original ResNet-18 + MLP for 12 epochs with TensorBoard and per-epoch `.rrd`/trajectory figures; select epoch 10 using validation only
- [x] Evaluate the frozen expert on 20 held-out episodes: 100% success, route completion 0.98499, collision/out-of-road 0
- [x] Finish the trained baseline's 20 held-out closed-loop episodes and latency acceptance; preserve interrupted run separately
- [x] Pass 26 tests, ROCm matmul/SDPA and the real native full-body off-road/grace probe; preserve planned collection handovers and failed probes
- [x] Validate native Rerun rendering and isolated desktop keyboard/manual simulator control; save video/rrd and distinguish automated events from human testing
- [x] Preserve Rerun screenshot-auto-exit panic and minimal reproducer (known unresolved viewer issue, not a claim of clean exit)
- [x] Complete stage 2 acceptance report, video, measured Marimo report/export and 26-test verification; stop for user confirmation
- [x] Obtain explicit stage 2 acceptance before stage 3 (user: “继续吧” after the completed report)

## Stage 3: deterministic latent prediction and VICReg

- [x] Fix model, data pairing, health estimators and acceptance definitions before training
- [x] Implement original 6–8M Conv-Stem/Transformer encoder and one-step residual predictor within the 15–30M total budget; functional SDPA MATH only
- [x] Verify episode-safe adjacent pairs, target gradients, regularization and position-content separation in rank estimates
- [x] Run real ROCm forward/backward/memory smoke and preserve any failures (0.761 GiB at smoke batch 16; formal batch remains 64)
- [x] Train on v2 train only; validation checkpoint selection; retain config/seeds/source/GOAL hashes
- [x] Evaluate held-out latent prediction vs copy-last with episode-cluster confidence intervals; channel std, covariance spectrum, content rank and pooled rank
- [x] Deliver TensorBoard, periodic actual predicted/true latent `.rrd`/figures, PCA, video and Marimo measured reports; ADE/FDE remain missing
- [x] Complete stage 3 report, disclose pooled-rank limitation and numerical regression/MRE, and stop before stage 4
- [x] Obtain explicit stage 3 acceptance before stage 4 (user: “继续下一阶段”)

## Stage 4: deterministic joint world/action model

- [x] Freeze data/history timing, model/loss, selection, ablation and acceptance protocol (docs/STAGE4_PROTOCOL.md)
- [x] Implement episode-safe 4-frame history and masked 4-frame world / 10×4 trajectory targets without filtering noise/recovery
- [x] Preserve spatial tokens in deterministic joint predictor; warm-start only from the project's stage 3 checkpoint (22,372,200 total parameters)
- [x] Verify target isolation, normalization, history/cache/reset timing, gradient paths and real ROCm memory/forward/backward (41 tests; batch-128 smoke 10.177 GiB joint / 5.764 GiB action-only allocator peaks)
- [x] Train joint and action-only fine-tuning runs with identical initialization/seed/data and validation-only checkpoint selection (both 12 epochs, selected epoch 12)
- [x] Finish the joint run's 12 formal epochs; validation selects epoch 12 (ADE/FDE 0.128401/0.217915 m); action-only selects epoch 12 (ADE/FDE 0.125191/0.212249 m)
- [x] Evaluate frozen checkpoints on complete test data and a matched ResNet comparison cohort, with raw paired errors and episode confidence intervals (9,600 frames, 22 episodes: Joint ADE 0.1305 m vs ResNet 0.1798 m, delta 95% CI [-0.0558, -0.0437] m)
- [x] Evaluate 20 held-out closed-loop seeds using the same simulator/controller and measure decision/full-pipeline latency; retain failures (Joint: 95% success, route 0.9698, P95 pipeline 14.396 ms; Action-only: 25% success, route 0.2971 due to latent stalling)
- [x] Deliver TensorBoard, periodic predicted/true trajectories and latents in Rerun/NPZ/figures, Marimo and acceptance videos (stage4_acceptance.mp4, dashboard.html, 4 RRD files audited)
- [x] Report all stage 4 gates and deliverables, then stop before stage 5 (REPORT.md, acceptance.json, delivery_checks.json, results_manifest.json; 14GB peak gate removed per user specification given 20GB VRAM and mmap loading; allocator peak 10.18 GiB within budget; all 7 gates passed)

## Training Infrastructure Upgrade: Hardware Auto-Detection, Calibration & Operator Selection

- [x] Freeze Stage 4 model architecture, checkpoints, and evaluation results (no structure/weight modifications)
- [x] Implement non-intrusive hardware profiling module (`mini_wa_jepa/hardware.py`: GPU, CPU, RAM, ROCm/CUDA, precision, attention backends)
- [x] Implement vendor-optimized operator selector and fallback context (`mini_wa_jepa/operators.py`: MATH baseline, FLASH, EFFICIENT, CUDNN, AUTO)
- [x] Implement training parameter recommendation and calibration (`mini_wa_jepa/calibration.py`: dynamic VRAM headroom, VICReg sample statistics constraints, precision, workers)
- [x] Implement finite-retry OOM recovery and emergency checkpoint preservation (`mini_wa_jepa/oom_recovery.py`)
- [x] Integrate backward-compatible config and training entry points (`configs/stage4.py`, `training/joint.py`, `scripts/train_joint.py`, `scripts/evaluate_joint.py`, `scripts/calibrate_hardware.py`)
- [x] Verified full verification plan: hardware profiling (RX 7900 XT 20GB, Ryzen 7800X3D), candidate batch trial (batch 128 peak 10.01 GiB), frozen checkpoint operator fidelity test (MATH vs AUTO max abs diff 2.86e-6 m, rel L2 2.29e-7, allclose: True), training throughput +30.4% (293.6 -> 382.8 FPS, peak VRAM 10.18 -> 9.07 GiB), OOM recovery unit tests and smoke train passed. See artifacts/hardware_upgrade/REPORT.md.

## Pedagogical Stratification: existing Marimo chapters (prior revision completed; new learning interface pending)

2026-10-19 user clarification supersedes the prior video-first presentation: stop making new videos for this task. Research and propose beginner-facing explorable lessons with direct manipulation, linked explanations and play/pause/step/scrub where appropriate. Earlier chapters teach their own concepts rather than all becoming prediction demos. The existing reactive chapters and driving videos do not satisfy this new teaching-interface requirement. Proposal: `docs/INTERACTIVE_LEARNING_DESIGN.md`; implementation remains pending and Stage 6 is not started.

- [x] Research primary examples and document per-chapter interactions, annotated Chinese code disclosure and honest replay/sandbox boundaries
- [ ] Implement and review one BEV lesson as the proposed interaction template before extending the remaining chapters

2026-10-18 teaching-workbench revision supersedes the original UI descriptions below: all six lessons now have beginner prerequisites/glossaries, step-by-step explanations, reactive experiments, actual annotated Chinese code, complete lesson source, exercises and read-only formal source panels. Controls live in top-level `mo.ui.dictionary` objects with separate consumer cells. Six standalone `app.run()` executions and Marimo checks pass; real Chromium events verify all 28 controls, six changed chart hashes and chapter-state retention. Full suite: 55 tests passed. Removed hardcoded demo performance claims; Chapter 4 reads measured reports, Chapter 5 remains unmeasured. BEV tutorial semantics/center aligned to the accepted schema. Original pages and evidence preserved under `artifacts/tutorial_rework_20261018`; details in `docs/LEARNING_WORKBENCH.md`.

- [x] Standardize architectural decoupling: each chapter contains independent models, losses, datasets, and configurations with permissible code duplication so each chapter can be copied/modified without side effects.
- [x] Created `chapters/README.md`: Master curriculum index, runnable commands, and architectural safety guidelines.
- [x] Implemented `chapters/chapter0_hardware/`:
  - `README.md`, `hardware.py`, `operators.py`, `calibration.py`, `oom_recovery.py`, `test_chapter0.py` (all tests passing)
  - `app.py`: Interactive Marimo notebook with hardware cards, operator benchmarks, VRAM donut charts, parameter sliders, and OOM state machine visualizer.
- [x] Implemented `chapters/chapter1_bev/`:
  - `README.md`, `semantic_bev.py` (coordinates conversion, 4-channel specifications, colorizer)
  - `app.py`: Interactive Marimo notebook with 4-channel breakdown, pixel math derivation, and coordinate conversion sandbox with real-time sliders.
- [x] Implemented `chapters/chapter2_baseline/`:
  - `README.md`, `config.py`, `model.py` (ResNet18Trajectory, masked L1 loss), `dataset.py`, `train.py`
  - `app.py`: Interactive Marimo notebook with ResNet-18 tensor flow, masked loss formula derivation, trajectory comparison plot, and causal confusion analysis.
- [x] Implemented `chapters/chapter3_latent_vicreg/`:
  - `README.md`, `config.py`, `model.py` (BEVEncoder, OneStepPredictor, LatentModel), `loss.py` (VICReg), `dataset.py`, `train.py`
  - `app.py`: Interactive Marimo notebook with latent world model intuition, residual zero-initialization rationale, VICReg variance/covariance matrix derivation, and token PCA visualization.
- [x] Implemented `chapters/chapter4_joint_world_action/`:
  - `README.md`, `config.py`, `model.py` (constant_turn_trajectory, BEVEncoder, OneStepPredictor, JointModel), `loss.py` (joint_loss), `dataset.py`, `train.py`
  - `app.py`: Interactive Marimo notebook with constant-turn kinematic prior sandbox (sinc formulas, speed/yaw sliders), spatial token preservation tensor flow, and Joint (96.98%) vs Action-Only (29.71%) vs ResNet-18 benchmark comparison.
- [x] Implemented `chapters/chapter5_stress_testing/`:
  - `README.md`, `degradation.py` (blindspot, cutout, boundary loss, frame dropout), `benchmark.py`
  - `app.py`: Interactive Marimo notebook with sensor degradation sandbox, original vs corrupted BEV diff heatmaps, trajectory robustness under corruption, and global degradation curves.

## Stage 5: BEV occlusion, boundary dropout and frame-loss stress testing (Completed; awaiting user acceptance)

2026-10-18 formal start: protocol frozen in `docs/STAGE5_PROTOCOL.md`, `configs/stage5.py`. Open loop: exactly 9,600 matched test anchors, three frozen formal models, clean plus nine degradation settings. Closed loop: clean plus medium/severe settings for each corruption, three models, six prespecified held-out seeds (126 episodes); disclose the smaller closed-loop cohort vs Stage 4. All 12 smoke combinations (16 samples, 3 models × clean/severe occlusion/boundary/dropout) completed; clean reproduction passed. Formal open-loop job started under `artifacts/restart_stage5/open_loop`; actual results remain provisional until completion and audit. No retraining or Stage 6 work authorized.

2026-10-18 structure scan: Stage 4 acceptance remains passed (all 7 gates). Chapter 5 contains prototype degradation operators and a benchmark, but no Stage 5 measured artifacts were found. Existing `tests/` suite passed 48 tests; it does not cover the new chapter entry points. See `docs/REFACTOR_SCAN.md` for confirmed integration gaps. The earlier statement that stage 5 was unauthorized is superseded by AGENTS.md.

- [x] Fix root chapter navigation: map visible labels to `ch0`–`ch5`; verified all six selections render the corresponding chapter through `app.run()` and `marimo check app.py` passes (2026-10-18)
- [x] Add formal Stage 5 checkpoint loading using the frozen production architecture (Joint/Action-only/ResNet); keep old chapter benchmark as historical prototype
- [x] Remove hardcoded Chapter 5 curves and untrained demo claims; show only explicitly labeled input-degradation experiments with formal robustness metrics still missing
- [x] Align chapter BEV channel semantics and pixel center with the accepted native four-channel schema
- [x] Update chapter commands for the `src/` directory layout and document the old benchmark's remaining checkpoint incompatibility

- [x] Implement parameterized BEV degradation suite: frozen front occlusion, static-boundary dropout, and causal periodic 1–3-frame hold; online/offline sources and episode isolation verified
- [x] Run matched open-loop degradation evaluation on 9,600 held-out test anchors across Joint, Action-only, and ResNet baselines (ADE / FDE vs degradation severity)
- [x] Run closed-loop stress evaluation on held-out seeds under online perception degradation (completion, collision, off-road rates)
- [x] Analyze critical edge cases: sudden obstacles, dynamic occlusions, and recovery maneuvers; capture prediction vs ground-truth visual evidence
- [x] Deliver Stage 5 report, comparison charts, RRD recordings, video manifest, and Marimo export; verify whether world prediction delivers measurable robustness gain

## Later stages — require preceding stage acceptance

- [x] Historical video sample under `artifacts/public_driving_demo_002` preserved; video-first interpretation superseded by the 2026-10-19 request for understandable interactive lessons

- [x] Clarify requested primary visualization: one shared scene/video showing all candidates, selected_id, actual execution and next-cycle replanning; record contract in `docs/PLANNING_VISUALIZATION.md` and active GOAL.md. Layout mockup is illustrative only; Stage 6/7 model work has not started.
- [ ] Implement primary planning-process view through RerunLogger during the authorized Stage 6/7 work; retain prediction/truth and latent diagnostics as supporting evidence

- [x] Stage 6: action-conditioned world prediction (all scientific and delivery gates passed; report delivered)
- [x] Stage 7 experiments and delivery complete: generation, risk-gain and planning scientific gates FAILED; retain frozen MSE; Stage 8 not started
  - [x] Freeze Stage 7 protocol (`docs/STAGE7_PROTOCOL.md`) and configuration (`configs/stage7.py`)
  - [x] Implement formal risk collector (`scripts/collect_stage7_risk.py`) and preflight helpers (`scripts/preflight_risk_collection.py`)
  - [x] Update risk collection unit tests (`tests/test_stage7_collection.py`)
  - [ ] Execute right-turn preflight and formal risk data collection (by Codex)
  - [ ] Conditional Flow Matching trajectory head training & diversity validation (in progress by Codex)
  - [ ] Action-conditioned risk prediction head training & calibration (AUROC/AUPRC)
  - [ ] 3-Arm closed-loop evaluation (48 episodes, independent seeds 1010000/1020000)
- [ ] Stage 8: full MetaDrive closed-loop acceptance
- [ ] Stage 9: transfer to 3D racing game (final project goal)


## Stage 5 measured report completion

Formal evaluation completed: 9,600 matched anchors × 3 models × 10 conditions; 126 closed-loop episodes. Protocol/delivery gates passed=True. Results, per-condition paired CIs, raw predictions, TensorBoard, RRD, actual rollout videos, acceptance video and measured dashboard are under `artifacts/restart_stage5/`. Lower absolute error and smaller degradation increments are evaluated separately; no universal robustness gain is presumed. See REPORT.md and summary.json. Stop here awaiting explicit Stage 6 confirmation.

## Stage 7 Codex review — experiments complete, scientific acceptance failed (2026-10-20)

User explicitly authorized Stage 7 and requested Gemini implementation through Relay, with Codex planning, independent acceptance and defect review. Stage 6 remains accepted; Stage 8 remains pending. The initial protocol draft was subsequently frozen before formal experiments; the appended deployment-scoring section is authoritative for the three-arm comparison.

- [x] Verify original hardware/backend and baseline parameter count: native ROCm 2.9.1+rocm6.4, approximately 20 GiB VRAM; Stage 4 has 22,372,200 parameters. Preserve hashes for 10 prior best checkpoints.
- [x] Reject two empty Relay SUCCESS results; identify native headless CLI command confirmation denial, retain an audit, and use Gemini file editing with Codex executing validation commands.
- [x] Correct nonexistent Stage 6 checkpoint path, remove unobserved execution-time steering interventions from risk protocol, and correct unused pytest dependency/NumPy boolean identity assertions. Collection tests: 8 passed.
- [x] Execute and independently audit train-only risk preflight 001: 26 branches, collision 10 positive/16 negative, off-road 0 positive/26 negative; all ego/scene errors and BEV differences exactly zero. Hashes and per-outcome censor masks agree with raw outcomes.
- [x] Revise train-only risk preflight to achieve actual off-road event coverage without changing simulator termination semantics or using geometric labels. Preflight 001 fails this coverage check and is preserved; right-turn preflight has 6/26 off-road positives.
- [x] Review and run Gemini Flow Matching implementation; complete risk, planning, matched closed-loop comparisons and interactive delivery. Experiments/delivery complete, scientific acceptance FAILED.

Review records: `artifacts/restart_stage7/codex_review/`. Raw preflight: `artifacts/restart_stage7/preflight_001/` and its log. No new videos produced.

2026-10-19 user supersedes the delegation arrangement: Codex resumes implementation and experiments while the user investigates Relay. Stage 7 remains the only authorized active stage. Fixed right-turn preflight completed 26 branches, with 6 actual off-road positives; all 17 collection/flow tests passed after correcting contextlib.nullcontext. Formal Flow cache contains 20,000/2,048/2,048 expert anchors; first 25-epoch seed completed. Formal two-part train risk collection and the second Flow seed started. All results remain provisional until full audit.

2026-10-20 ongoing audit: both 25-epoch Flow runs completed; non-oracle test ADE 2.0370/1.7211m vs frozen MSE 0.1290m, so generation precision fails. No test-driven retraining or candidate postprocessing. Raw predicted diversity alone does not establish feasibility. Full suite initially 105/105 passed; real browser keyboard sliders, tab switching and same-record next-decision replay passed. Final suite includes later acceptance and observed-motion regression checks. Risk collection preserves typed native outcomes, currently only sidewalk crashes, with no vehicle collision evidence.

- [x] Find collector placeholder-ego defect before formal risk training. Preserve original raw files and zero-ego smoke artifacts. Implement separate exact prefix replay recovery with BEV/full-scene checks and immutable sidecars; formal caching refuses missing recovery.
- [x] Preserve native strict-replay failures in original risk_train_b/risk_test: each has one differing SIDEWALK/TERRAIN boundary pixel with identical float64 poses/camera/state. Fresh-engine and 24-bit-depth probes fail to eliminate it; camera-only fixed semantic draw order passes 16/16 repeats at each failed context. No pixel masking or tolerance relaxation. Source diff and exact version allowlist are retained in codex_review/determinism_repair.*.
- [x] Add continuation into new partitions, copying only completed contexts and recomputing counts; old failed partitions/partial branches stay untouched. All original map/seed/anchor contexts remain required. Each context records its renderer version, and motion replay uses that version. Source collector now writes actual ego history rather than a placeholder.
- [x] Finish 192-context risk collection/recovery, matched world/zero-world heads, two-seed three-arm closed loop (48 episodes), report, RRD readback and final interactive delivery audit.
- [x] Stop after Stage 7 report. Stage 8 requires explicit user confirmation and has not started.

Final Stage 7: Flow sample1 ADE 2.0370/1.7211m vs MSE0.1290m; pointwise kinematic validity52.91%/54.33% fails95%. Risk test has256crash/28offroad positives across8/4 positive seed clusters; all crashes are sidewalk contacts, with no vehicle collisions. World improves held-out offroad Brier42.46%/34.85%, but seed20261021 CI crosses zero (also crash gain CI), so the frozen two-seed world-gain criterion fails. All48 closed-loop episodes completed: MSE mean route0.985084 on both paired repeats; geometry/full0.015826, fallback fraction1.0 and mean actual travelled distance only0.0825m (MSE319.93m). The small route fraction mainly reflects the spawn offset, not learned driving. Retain the original MSE policy; do not start Stage8 or retune on these test outcomes.

Delivery:111 tests passed,142 native RRD files read successfully, actual browser tab/keyboard-slider/replay checks passed,56 measured dashboard rows without parser errors, Chinese Chapter7 and static HTML snapshots delivered. Source/config/seeds/GOAL hashes, all raw arrays, checkpoints, TensorBoard, latent/trajectory figures and RRD, prior failures/MREs and source-repair audit retained. All10 preexisting best.pt hashes unchanged. Reports: `artifacts/restart_stage7/REPORT.md`, summary.json, acceptance.json, metrics.json; technical review: codex_review/review_findings.md. Static exports are snapshots; root `app.py` provides runtime interactivity. No new video or known-broken desktop path was launched.

The Stage7 Flow feature build recreated the rebuildable `outputs/data/stage2_v2_100k/baseline_cache` (~7.5GB) for aligned expert labels and MSE controls. The earlier cache-cleanup inventory records its historical deletion, not the current rebuilt state. Schema1 training datasets remain removed.

Report-only audit failures were preserved and corrected: repeat determinism now compares predictions/labels/indices and records varying timing arrays separately; four early process-monitor records missing PSS remain explicitly missing and are never substituted with RSS or zero. No experiment was retrained or reselected for either reporting fix.
