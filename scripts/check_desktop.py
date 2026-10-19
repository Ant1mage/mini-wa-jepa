"""Stage 1 supplement: native use_render=True AMD desktop window benchmark."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from env.expert import NavigationExpert
from env.stage2 import make_data_env
from evaluation.visuals import main_view, prepare_graphics


def main():
    _, restore = prepare_graphics(False)
    sim = make_data_env("SC", 9000, render_mode="live")
    timings = []
    root = Path("artifacts/stage1/desktop")
    root.mkdir(parents=True, exist_ok=True)
    try:
        sim.reset(seed=9000)
        expert = NavigationExpert(sim)
        drawn=[]
        def count_frame(task):
            drawn.append(time.perf_counter())
            return task.cont
        sim.engine.taskMgr.add(count_frame,'stage1_draw_counter',sort=60)
        for step in range(400):
            start = time.perf_counter()
            _, _, tm, tc, _ = sim.step(expert.act()[0])
            timings.append(time.perf_counter()-start)
            if step == 200:
                Image.fromarray(main_view(sim)).save(root / "native_window.png")
            if tm or tc:
                break
        g = sim.engine.win.getGsg()
        report = dict(use_render=sim.config["use_render"], window_open=sim.engine.win.getProperties().getOpen(),
            pipe=sim.engine.pipe.getType().getName(), renderer=g.getDriverRenderer(), vendor=g.getDriverVendor(),
            version=g.getDriverVersion(), steps=len(timings),
            control_steps_per_second=len(timings)/sum(timings),
            native_rendered_frames=len(drawn),
            native_render_fps=(len(drawn)-1)/(drawn[-1]-drawn[0]),
            rendered_frame_ms_p50_p95=(np.percentile(np.diff(drawn),[50,95])*1000).tolist(),
            frame_ms_p50_p95=np.percentile(timings, [50,95]).__mul__(1000).tolist(),
            decision_dt_s=0.1, native_window_size=list(sim.config["window_size"]),
            multi_thread_render=sim.config['multi_thread_render'],sync_video=False,
            note="Native onscreen window. Explicit render-task FPS. ForceFPS disabled; viewer owns pacing, simulation can run faster than wall time.")
        assert report["window_open"] and "7900 XT" in report["renderer"]
        (root / "report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        sim.close()
        restore()


if __name__ == "__main__":
    main()
