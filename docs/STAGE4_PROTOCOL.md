# Stage 4 protocol — fixed before formal training

The user's “继续下一阶段” accepts stage 3 and authorizes stage 4 only. Stop at the stage 4 report. No occlusion, dropout, generative head or stage 5 business logic is introduced.

## Inputs, labels and timing

Use the accepted v2 100k/10k/10k dataset and its train-only trajectory normalization. At every anchor use four observations at t−0.3/−0.2/−0.1/t seconds, each with speed and yaw rate. History never crosses episode resets or step gaps; unavailable startup history repeats the first observation in that contiguous segment. Yaw rate uses only past headings. Predict four actual future encoder token arrays at +0.1/+0.2/+0.3/+0.4s and ten ego-frame `(x,y,v,heading)` waypoints at +0.2…+2.0s. Mask missing future frames individually. Keep noisy/recovery action supervision and valid partial trajectories; terminal/missing labels use the existing v2 masks, not a new quality filter.

All anchors remain available. Report open-loop ADE/FDE only on the same action-valid complete 2s trajectories used by the stage 2 ResNet baseline, including startup anchors with padded history. Verify identical raw anchor IDs for paired comparisons. Invalid targets never contribute to world prediction statistics or VICReg target regularization.

## Original deterministic architecture

Reuse only this project's stage 3 epoch-10 selected weights. The 6,213,952-parameter encoder preserves 8×8 spatial tokens. At each spatial location concatenate the three past-current token differences and the observed four-frame motion history, then learn a residual fusion into the current token. A six-layer d=384 functional SDPA MATH predictor emits four future token residuals in one pass. Initialize its four output slices by multiplying the stage 3 one-step output by horizon index 1…4. This is a trainable initial extrapolation, not an assumption that future targets are known.

The trajectory head projects the current and four predicted latent arrays per spatial location, preserves all 64 locations by flattening, then uses an MLP. It does not average spatial tokens into the low-rank global representation found in stage 3. Total parameters: 22,372,200; encoder 6.214M, world predictor plus temporal fusion 11.512M, remaining action head 4.646M. This is within the 15–30M total and 8–12M predictor budgets.

Use an explicit constant observed speed/yaw-rate arc as the trajectory head's initial kinematic prior, plus a learned residual in training-normalized `(x,y,v,heading)` units. It uses no navigation, future action, future ego state or route labels. The same prior/history/architecture applies to the action-only control. This differs from the stage 2 single-frame ResNet; improvements over ResNet alone cannot be attributed exclusively to a world loss.

## Training and ablation

Joint objective: `25*MSE_world + 25*VICReg_variance + VICReg_covariance + 100*masked_SmoothL1_trajectory`. Both online/future encoder branches receive gradients. Compute covariance after subtracting each fixed spatial position's batch mean, as in stage 3. Gamma=1 and epsilon=1e−4. AdamW, 12 epochs, batch 128, seed 20261018, cosine schedule; encoder/predictor/action learning rates 3e−5/1e−4/3e−4. Use bf16 training, FP32 evaluation, SDPA MATH, no compile or external model/weights. Clip gradients at 5 with allocator peak within hardware 20GB VRAM budget and dataset mmap loading, retaining process DRM/RSS measurements.

Run the matched action-only fine-tuning control with latent_weight=0, disabling all world prediction/variance/covariance losses. It uses the same architecture, initialization, seed, batch order, epochs, kinematic prior, controller and validation checkpoint rule. Gradients still reach predicted latent features through the action head. Both runs retain stage 3 pretraining, so this control isolates the effect of the joint-training latent objective, not removal of every historical world prior. No causal claim about removing pretraining is made.

Checkpoint selection is minimum ADE on a fixed seed-selected set of 4096 validation anchors. Training never evaluates test. Select each run independently by this rule; evaluate the frozen selected checkpoints on all test anchors once. For the same complete-trajectory anchors, recompute frozen ResNet errors and save raw per-anchor ADE/FDE/predictions, with paired episode-cluster bootstrap intervals (2000 samples) for joint−ResNet and joint−action-only. These intervals are descriptive; ADE/FDE acceptance uses the actual point estimates specified in GOAL.md.

Stop training and preserve a failing batch/model/config MRE on nonfinite values, OOM or severe content degeneration (effective rank <2 or mean std <0.01). A missed gate is reported as a failure, not permission to tune on test or silently restart.

## Closed loop and latency

Use the same MetaDrive environments, traffic density, off-road grace rule, pure-pursuit/PID tracker and 20 held-out seeds as stage 2: SCS/309000–309009 and CSC/319000–319009. No action noise or observation degradation is injected. Physics and control dt=0.02s. Clear all history on reset. Retain past encoder outputs so history frames at exact five-control-step intervals need not be re-encoded; encode the current observation each step. The model always uses observed history, never recorded expert future states. Check cached and full-history inference equivalence in tests.

Padding at startup repeats the first observation. Past-interval yaw rate follows the 0.1s training definition once enough history exists; startup uses the available past interval. Controller speed clipping/feedforward and lateral lookahead are identical to stage 2. Do not use an expert fallback to hide failures.

Report decision latency (state/history, CPU→HIP encoding, prediction, inverse normalization and tracking) separately from the full pipeline, which additionally includes env.step/native semantic rendering/readback/four-channel conversion. Exclude reset and visualization/disk encoding, and the first 20 steps per episode, matching the stage 2 measurement contract. Report cold resets separately. Latency acceptance is full-pipeline P95 <=25ms, with raw per-step samples retained. Run latency evaluation without concurrent GPU training.

Acceptance: full test ADE and FDE both strictly below the frozen stage 2 baseline; mean closed-loop route completion >=90% of its 0.985049643944109 score (>=0.8865446795496981); full-pipeline P95 <=25ms; parameters and training memory within budget. Also disclose collision/off-road/success rates, lateral errors, speed, representation health, inference percentiles and any deterioration in the ablation. These gates do not validate stage 5 robustness.

## Evidence

Every 400 optimizer steps and at epoch end, use fixed validation IDs (0,512,5500,8000), covering both val map families, and visualization seed 20261018. Save actual predicted/true trajectories, valid future latent arrays and masks to NPZ and independent Rerun recordings; align optimizer step with TensorBoard and separately record simulation time, episode/map/seed/horizon and startup padding. Keep predicted-vs-truth PNGs, world/copy error, content/pooled rank and covariance diagnostics. No latent heatmap is called a pixel reconstruction.

Record representative closed-loop episodes for both policies on each test map with actual third-person RGB, BEV, model trajectories and delayed realized trajectory/latent targets. Save actual raw arrays so missing tail targets remain masked. Deliver videos, raw metrics/timing, provenance/config/seeds/GOAL hash/source snapshots, stage report and Marimo open/closed-loop and ablation tables. Preserve all stage 1–3 reports, retired-data inventories, and the known Rerun screenshot-auto-exit failure.
