# Preserved viewer probes

These probes concern independent Rerun processes only. No same-process Panda3D/Pygame rendering was relaunched.

1. The original SDK `rr.spawn(recording=stream)` selected PATH's Rerun 0.39.0 while the project SDK was 0.22.1. Observed warning:

   `The version of the Rerun Viewer available on your PATH does not match the version of your Rerun SDK`

   `Rerun Viewer: v0.39.0 (executable: "rerun") / Rerun SDK: v0.22.1`

   The first receiver probe made an empty TCP readiness connection, causing `--test-receive` to exit with `Error: EntityDb never initialized`. The `python -m rerun` 0.22 entrypoint returned 0 despite this binary error. Preserve stream_probe.json and stream_receiver.log as failed probe evidence; they are not successful streaming validation.

2. The first bundled-viewer adapter incorrectly selected experimental `connect_grpc` merely because the 0.22 SDK exposed it. Observed:

   `ERROR re_grpc_client::message_proxy::write: Write messages call failed: status: Unknown, message: "transport error"`

   Its file sink is retained as spawn_matched_sdk.rrd. This is a failed transport probe.

3. The corrected adapter starts the SDK-bundled executable on a separate port and selects TCP for 0.22 / gRPC for 0.39. The five-step 0.22 startup/file probe then completed without the transport error; spawn_matched_tcp.rrd is retained. A separate headless receiver probe used the bundled binary directly and waited for LISTEN without opening a dummy connection; stream_probe_fixed.json records successful data reception and binary exit 0. Its sent_stream_fixed.rrd and stream_receiver_fixed.log are retained.

4. Native screenshot probes exited 0 but did not create their requested PNGs. Desktop frame content and keyboard interaction therefore remain unverified. Exact minimal reproduction:

   `.venv/bin/rerun artifacts/visualization_standard/run_001/replay_legacy.rrd --screenshot-to artifacts/visualization_standard/run_001/rerun_matched.png --window-size 1400x1000`

   SDK/viewer 0.22.1, DISPLAY=:0, WAYLAND_DISPLAY=wayland-0. See rerun_matched.log (empty) and verification.json. No GPU/backend override was used. No claim is made that the prior native display failures are fixed.
