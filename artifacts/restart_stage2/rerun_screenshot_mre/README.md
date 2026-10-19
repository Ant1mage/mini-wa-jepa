# Rerun 0.22.1 screenshot shutdown MRE

Run from the repository root, with a new screenshot path:

```bash
.venv/bin/python -m rerun --screenshot-to /tmp/rerun-mre.png --window-size 800x600 artifacts/restart_stage2/rerun_screenshot_mre/minimal.rrd
```

`minimal.rrd` is an 86KB recording generated through the public `RerunLogger`: one time step, one uint8 `(4,128,128)` BEV, and ego state. No simulator is started by this reproduction.

Observed on the current desktop: the native Wayland viewer renders and writes `screenshot.png`, then panics with `Failed to take store hub from the Viewer` at `re_viewer/src/app.rs:1851`. The Python CLI wrapper returned 0 despite the panic, so success cannot be inferred from its exit code. `viewer.log` retains the backtrace. The same failure occurred with the full expert recording; its screenshot and log are in the parent directory. This issue is not marked fixed and no hardware-specific workaround was applied.
