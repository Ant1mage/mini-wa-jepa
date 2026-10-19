"""Run in a fresh process for each backend; always close the MetaDrive engine."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from configs.stage1 import Stage1Config
from env.verification import capture_bev, capture_rgb, make_env, state_snapshot


def validate_image(image, shape):
    assert image.shape == shape, (image.shape, shape)
    assert image.dtype == np.uint8, image.dtype
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] >= 3, "Empty/constant image"
    return dict(shape=list(image.shape), dtype=str(image.dtype),
                mean=float(image.mean()), std=float(image.std()),
                unique_colors=int(np.unique(image.reshape(-1, 3), axis=0).shape[0]))


def contact_sheet(paths, out):
    if not paths:
        return
    sheet = Image.new("RGB", (5 * 256, ((len(paths) + 4) // 5) * 280), "#222222")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((256, 256))
        x, y = (i % 5) * 256, (i // 5) * 280
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + 258), path.stem, fill="white")
    sheet.save(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["default", "egl", "software"], default="egl")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--capture-count", type=int, default=20)
    p.add_argument("--egl-device", type=int, default=-1)
    p.add_argument("--msaa-limit", type=int, choices=[0, 2, 4, 8])
    p.add_argument("--output", type=Path, default=Path("artifacts/stage1/egl"))
    args = p.parse_args()
    # Clear both display variables to prove no desktop/display server is needed.
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    if args.backend == "software":
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    from panda3d.core import loadPrcFileData
    if args.backend in ("egl", "software"):
        loadPrcFileData("", "load-display p3headlessgl")
        loadPrcFileData("", f"egl-device-index {args.egl_device}")
    if args.backend == "software":
        loadPrcFileData("", "framebuffer-hardware false\nframebuffer-software false")
    loadPrcFileData("", "audio-library-name null\nnotify-level error")
    cfg = Stage1Config(episodes=args.episodes, horizon=args.horizon,
                       start_seed=args.start_seed, capture_count=args.capture_count)
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in ("bev", "rgb"):
        (args.output / kind).mkdir(exist_ok=True)
    report = dict(config=asdict(cfg), backend=args.backend, seed=cfg.start_seed,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  metadrive_commit=subprocess.check_output(["git", "-C", ".deps/metadrive", "rev-parse", "HEAD"], text=True).strip(),
                  display=os.getenv("DISPLAY"), wayland_display=os.getenv("WAYLAND_DISPLAY"),
                  libgl_always_software=os.getenv("LIBGL_ALWAYS_SOFTWARE"),
                  egl_device_index=args.egl_device, msaa_limit=args.msaa_limit,
                  episodes=[], image_checks=[], completed_without_exception=False)
    restore = lambda: None
    if args.msaa_limit is not None:
        from env.render_compat import limit_filter_msaa
        report["msaa_adjustments"], restore = limit_filter_msaa(args.msaa_limit)
    sim = None
    total_steps = 0
    step_seconds = bev_seconds = rgb_seconds = 0.0
    started = time.perf_counter()
    capture_seeds = set(np.linspace(cfg.start_seed, cfg.start_seed + cfg.episodes - 1,
                                    min(cfg.capture_count, cfg.episodes), dtype=int).tolist())
    try:
        sim = make_env(cfg)
        for seed in range(cfg.start_seed, cfg.start_seed + cfg.episodes):
            reset_started = time.perf_counter()
            sim.reset(seed=seed)
            reset_seconds = time.perf_counter() - reset_started
            if not report.get("graphics"):
                gsg = sim.engine.win.getGsg()
                report["graphics"] = dict(vendor=gsg.getDriverVendor(), renderer=gsg.getDriverRenderer(),
                                          version=gsg.getDriverVersion(), pipe=sim.engine.pipe.getType().getName())
                report["initial_state"] = state_snapshot(sim)
            camera = sim.engine.get_sensor("rgb_camera")
            assert camera.cam.getParent() == sim.agent.origin, "RGB camera is not attached to ego"
            captured = False
            for step in range(1, cfg.horizon + 1):
                tick = time.perf_counter()
                _, _, terminated, truncated, info = sim.step([0, 0])
                step_seconds += time.perf_counter() - tick
                total_steps += 1
                # Extract both sensors every step, including the 80 non-saved episodes.
                tick = time.perf_counter()
                bev = capture_bev(sim, cfg)
                bev_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                rgb = capture_rgb(sim)
                rgb_seconds += time.perf_counter() - tick
                if seed in capture_seeds and not captured and (step >= 100 or terminated or truncated):
                    checks = dict(seed=seed, step=step,
                                  bev=validate_image(bev, (cfg.bev_size, cfg.bev_size, 3)),
                                  rgb=validate_image(rgb, (cfg.rgb_height, cfg.rgb_width, 3)),
                                  camera_parent_is_ego=bool(camera.cam.getParent() == sim.agent.origin),
                                  effective_bev_pixels_per_metre=float(sim.engine.top_down_renderer.scaling))
                    report["image_checks"].append(checks)
                    for kind, img in (("bev", bev), ("rgb", rgb)):
                        Image.fromarray(img).save(args.output / kind / f"seed_{seed:03d}_step_{step:03d}.png")
                    captured = True
                if terminated or truncated:
                    break
            assert terminated or truncated, "Horizon did not produce a Gymnasium episode boundary"
            row = dict(seed=seed, steps=step, terminated=bool(terminated), truncated=bool(truncated),
                       reset_seconds=reset_seconds,
                       **{k: info.get(k) for k in ("arrive_dest", "crash", "crash_vehicle", "crash_object",
                                                   "crash_building", "out_of_road", "max_step", "route_completion")})
            report["episodes"].append(row)
            if len(report["episodes"]) % 5 == 0 or cfg.episodes < 5:
                print(json.dumps(row), flush=True)
            (args.output / "progress.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
        report["final_state"] = state_snapshot(sim)
        report["completed_without_exception"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if sim is not None:
            sim.close()
        restore()
        elapsed = time.perf_counter() - started
        report.update(total_steps=total_steps, elapsed_seconds=elapsed,
                      step_seconds=step_seconds, bev_seconds=bev_seconds, rgb_seconds=rgb_seconds,
                      steps_per_second_step_only=total_steps / step_seconds if step_seconds else 0,
                      steps_per_second_with_extraction=total_steps / (step_seconds + bev_seconds + rgb_seconds)
                      if total_steps else 0,
                      steps_per_second_end_to_end=total_steps / elapsed)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    for kind in ("bev", "rgb"):
        contact_sheet(sorted((args.output / kind).glob("*.png")), args.output / f"{kind}_contact_sheet.png")
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "image_checks")}, indent=2, default=float))


if __name__ == "__main__":
    main()
