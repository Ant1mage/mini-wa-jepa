# Stage 7 delegation audit (in progress)

Stage 6 remains accepted. This document records Stage 7 implementation/review issues; it is not a Stage 7 completion report.

## Runtime issues verified by Codex

1. First read-only dispatch failed before execution: `Antigravity headless cannot enforce a read-only workspace; Relay refuses to run this access mode`. No review output was produced. A new authorized write task was used for actual protocol/code deliverables, rather than claiming read-only enforcement.
2. Write task iteration 1 reported `SUCCESS`, exit 0 and an empty summary, without any requested file. Relay events show the last operation was `run_command: ls -la docs/`, then a null tool result. Codex rejected completion.
3. Iteration 2 again reported `SUCCESS`, exit 0 and empty summary, without any requested file. This time the last operation was `run_command: ls -F docs/`. Codex rejected completion again. Native CLI log identifies `Print mode: soft-denying tool confirmation "RunCommand" at step 16`, followed by input closure. Thus the CLI success status does not establish actual task completion; the terminal call did not receive approval in that runtime.
4. Iteration 3 was instructed to avoid terminal tools and directly write only two protocol/config files. Codex can run authorized commands in the primary workspace for subsequent independent validation. No global permission settings were changed and no permission check bypassed.

Evidence IDs: run `f309bd58-feab-49b7-9e24-2a0ba23ae397`; first write worker `5c8d69ba-0451-4a8d-84d8-2f026c738369`; second `14289b08-d5e0-4c27-825a-847e28672558`; third `61d2de22-05ba-4e4e-ae0b-c9e8c46a9e28`. Read-only failed worker `ea6ef2cc-31f5-4c84-844e-0ae0195a528c`.

## Scientific review concerns to resolve before formal runs

- Stage 6's zero observed risk labels cannot validate Stage 7 risk prediction. Require real positive/negative outcomes and explicit per-outcome censoring.
- Stage 6 future horizons 0.5/1/1.5/2s differ from Stage 4's 0.1/0.2/0.3/0.4s. Do not run the unchanged Stage 4 action head on Stage 6 latent predictions and call it the frozen MSE baseline.
- Count deployed parameters with a shared encoder rather than duplicating both entire models unnoticed. Retain separate frozen MSE baseline for its evaluation arm.
- Collector may use teacher route to propose experimental commands; deployed planner must score from available observations, without privileged future route/traffic information.

## Implementation defects found and corrected / pending

- Initial Stage 7 config referenced missing `artifacts/restart_stage6/conditioned_20261019/best.pt`; corrected to the existing checkpoint under `runs/` after Codex checked the filesystem.
- Initial proposed risk campaign added execution-time steering bias and delayed braking invisible to a trajectory-conditioned model. Codex rejected this confound; subsequent candidates execute only through the unchanged trajectory tracker.
- Initial tests imported unavailable pytest despite no usage, and used `mask[i] is True/False` on NumPy booleans. Gemini converted them to standard unittest and corrected assertions. Codex ran all 8 tests successfully.
- The train-only preflight completed 26 branches with strict zero replay differences. Independent raw audit: crash 10 positive/16 negative, off-road 0 positive/26 negative. Coverage failed; successful execution is not sufficient risk supervision. Preserve preflight 001 and revise observed initial contexts rather than changing the native off-road grace rule or fabricating labels.
