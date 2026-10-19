# GOAL.md restart analysis

The newly added front section is the current plan. The older plan remains appended; it is preserved as history rather than combined into contradictory acceptance rules. Current instruction: **stage 1 only**. Earlier desktop/live/video requests still apply.

| Item | Previous implementation | Current requirement |
|---|---|---|
| BEV | 64x64 CHW, 3 channels; 256px color diagnostic | 128x128 CHW, 4 channels including visibility, 0.5m/pixel |
| Action | 25 (x,y,heading) points at 0.1s | 10 (x,y,v,heading) points at 0.2s |
| Representation | EMA/deterministic then world flow planned | Conv-Stem + Transformer, deterministic + VICReg |
| Parameters | 20–40M | 15–30M; <=14GB training memory |
| Stage 2 | first 5k frames; no neural baseline | 100k recovery data + ResNet-18 + closed loop |
| Action labels | excluded large lateral deviation / short horizons | retain disturbance/recovery; exclude unrecoverable terminal failure |
| Main experiment | general world/action demo | partial observability and sensor dropout |
| Environment gate | 100 episodes, sample images | 100 episodes, VRAM monitoring, >=120 complete-pipeline FPS |

## Reuse and limitations

The existing virtual environment, official pinned MetaDrive/assets, ROCm/SDPA MATH checks and verified simulator APIs remain useful. Previous EGL results (100 episodes, 34,494 steps, 96.74 steps/s with BEV/RGB extraction) do not validate the new four-channel input or memory-leak requirement. The 59.4 FPS native desktop test is also not the >=120 headless gate.

Old stage 2 produced 5k three-channel frames and a recovery video. Preserve them; do not mix them into a new dataset. Same-process Panda3D/Pygame tests failed with SIGSEGV and GLX BadAccess. The prior commit title claiming a “stable” viewer was premature: the alternative failed too. No successful fix is claimed.

## Stage 1 definitions

Use native Panda3D SemanticCamera with an orthographic 64m film. Check independently the lens projection of known ego-relative points. The un-degraded ground-truth mode has visibility=255 throughout its footprint: this explicitly denotes full visibility, not an implemented ray-casting degrader. Stage 5 logic is not written ahead of its gate.

Measure env.step/render/readback/semantic-channel extraction together. Also report reset-inclusive wall throughput. Never use a fast sub-operation as complete-pipeline FPS. Record process DRM VRAM when supported, device-wide VRAM and RSS, plus source/config/goal hashes. Report warmed memory ranges and growth slopes; a finite run can show no observed leak, not prove lifetime leak freedom. Failed thresholds leave the stage incomplete.

## Later-stage issues to resolve at their gates

- Existing 0.1s decisions are 10Hz; >=40Hz control needs a separate control cadence and 0.2s waypoint sampling. Neural latency alone does not establish control frequency.
- Covariance from only B pooled samples has rank <=B-1. A d=256 effective-rank target >=192 needs enough evaluation observations and a defined token/time aggregation; positional variation must not inflate the health claim.
- Two stride-2 convolutions leave 1024 tokens/frame. Joint attention over history can be expensive with SDPA MATH. <=25ms / <=14GB must be profiled, not promised from parameter count.
- Single-frame BC versus temporal JEPA changes memory and objective. The no-world-loss ablation must retain history to isolate the world objective.
- “10% disturbance probability” needs a trigger unit (per step or per episode/segment); this changes the perturbed-data fraction.
