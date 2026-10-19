"""Shared RGB + BEV + trajectory + status layout, for live, replay and video."""
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from mini_wa_jepa.geometry import ego_to_pixels


def prepare_graphics(headless=False, offscreen_egl=False):
    import os
    from panda3d.core import loadPrcFileData
    if headless:
        os.environ.pop("DISPLAY", None)
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    if headless or offscreen_egl:
        loadPrcFileData("", "load-display p3headlessgl\negl-device-index 1")
    loadPrcFileData("", "audio-library-name null\nsync-video false")
    from env.render_compat import limit_filter_msaa
    return limit_filter_msaa(8)


def main_view(sim):
    # Native MainCamera.perceive CPU arrays are BGR, as verified in its source.
    return np.ascontiguousarray(sim.main_camera.perceive(to_float=False)[..., ::-1])


def bev_rgb(bev):
    if bev.shape[0] == 4:
        from mini_wa_jepa.visualization import colorize_bev_4ch
        return colorize_bev_4ch(bev)
    rgb = np.zeros((*bev.shape[-2:], 3), dtype=np.uint8)
    rgb[:] = (20, 26, 35)
    rgb[bev[0] > 0] = (80, 110, 100)
    rgb[bev[1] > 0] = (245, 178, 55)
    rgb[bev[2] > 0] = (235, 70, 95)
    return rgb


def compose(bev, trajectory, state, action, status, camera=None, caption="expert plan", extent_m=64):
    canvas = np.full((640, 1280, 3), (17, 21, 29), dtype=np.uint8)
    if camera is not None:
        canvas[40:445, :720] = cv2.resize(camera, (720, 405))
    else:
        cv2.putText(canvas, "No RGB stored: BEV-only dataset", (25, 160), 0, 0.7, (210, 215, 225), 1)
    panel = cv2.resize(bev_rgb(bev), (512, 512), interpolation=cv2.INTER_NEAREST)
    if trajectory is not None and len(trajectory):
        size = bev.shape[-1]
        pts = ego_to_pixels(np.asarray(trajectory)[..., :2], size, extent_m)
        pts = np.rint((pts + 0.5) * (512 / size)).astype(np.int32)
        cv2.polylines(panel, [pts], False, (40, 220, 250), 2)
        for point in pts[::2]:
            cv2.circle(panel, tuple(point), 3, (40, 220, 250), -1)
    cv2.arrowedLine(panel, (256, 270), (256, 246), (255, 255, 255), 3, tipLength=0.4)
    canvas[40:552, 748:1260] = panel
    text = ["MetaDrive / third person", f"BEV ({bev.shape[0]} channels) | {caption}",
            f"speed {state[3]:.2f} m/s  steer {action[0]:+.3f}  throttle/brake {action[1]:+.3f}",
            status, "SPACE pause/resume | N step | R reset | Q quit | M manual/auto | W A S D drive"]
    for label, xy, scale in zip(text, [(15,28),(750,28),(20,480),(20,520),(20,610)], [.65,.45,.64,.57,.54]):
        cv2.putText(canvas, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (225,230,240), 1, cv2.LINE_AA)
    return canvas


class LiveViewer:
    def __init__(self, title="mini-wa-jepa | expert", sim=None, fps=30):
        import pygame
        self.pg, self.sim = pygame, sim
        pygame.display.init()
        self.screen = pygame.display.set_mode((1280,640))
        pygame.display.set_caption(title)
        self.clock, self.fps = pygame.time.Clock(), fps
        self.paused = self.manual = self.quit = self.reset_requested = self.single_step = False
        # Bind the same controls in the native Panda3D window as well.
        self.held = set()
        if sim is not None:
            for key in ("space", "n", "r", "q", "m"):
                sim.engine.accept(key, self.key, [key])
            for key in ("w", "a", "s", "d"):
                sim.engine.accept(key, self.held.add, [key])
                sim.engine.accept(key+"-up", self.held.discard, [key])

    def key(self, key):
        if key == "space": self.paused = not self.paused
        elif key == "n": self.single_step = True; self.paused = True
        elif key == "r": self.reset_requested = True
        elif key == "q": self.quit = True
        elif key == "m": self.manual = not self.manual

    def poll(self):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT: self.quit = True
            elif event.type == self.pg.KEYDOWN:
                name = self.pg.key.name(event.key)
                self.held.add(name)
                self.key(name)
            elif event.type == self.pg.KEYUP:
                self.held.discard(self.pg.key.name(event.key))

    def draw(self, frame):
        self.poll()
        self.pg.surfarray.blit_array(self.screen, np.transpose(frame, (1,0,2)))
        self.pg.display.flip()
        self.clock.tick(self.fps)

    def ready(self):
        self.poll()
        if self.sim is not None:
            self.sim.engine.taskMgr.step()  # process native keys even while paused
        if self.single_step:
            self.single_step = False
            return True
        return not self.paused

    def manual_action(self):
        return np.array([float("a" in self.held)-float("d" in self.held),
                         float("w" in self.held)-float("s" in self.held)], dtype=np.float32)

    def close(self):
        if self.sim is not None:
            for key in ("space","n","r","q","m","w","a","s","d","w-up","a-up","s-up","d-up"):
                self.sim.engine.ignore(key)
        self.pg.display.quit()


class VideoWriter:
    def __init__(self, map_id, seed, stage="stage2", fps=10):
        import imageio_ffmpeg
        timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S_%f")
        self.path = Path("outputs/videos") / stage / f"map-{map_id}_seed-{seed}_{timestamp}.mp4"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x640", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("Video encoder failed")
