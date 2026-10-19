"""Exercise the isolated desktop keyboard through X11 events alongside EGL.

This is an automated desktop integration probe, not a claim of human testing.
Only the probe's own control window receives injected keys.
"""
import argparse
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import time

import numpy as np

from configs.stage2 import Stage2Config
from env.data_v2 import NativeBEVProvider, configure_egl, make_env, third_person
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from mini_wa_jepa.visualization.keyboard import KeyboardControls


class DesktopKeys:
    def __init__(self, window_id):
        self.x = C.CDLL(ctypes.util.find_library('X11'))
        self.x.XOpenDisplay.argtypes = [C.c_char_p]
        self.x.XOpenDisplay.restype = C.c_void_p
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('Desktop X11 connection unavailable')
        self.x.XDefaultRootWindow.argtypes = [C.c_void_p]
        self.x.XDefaultRootWindow.restype = C.c_ulong
        self.x.XStringToKeysym.argtypes = [C.c_char_p]
        self.x.XStringToKeysym.restype = C.c_ulong
        self.x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        self.x.XKeysymToKeycode.restype = C.c_ubyte
        self.x.XSync.argtypes = [C.c_void_p, C.c_int]
        self.x.XCloseDisplay.argtypes = [C.c_void_p]
        self.x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.c_void_p]
        if not window_id:
            raise RuntimeError('SDL did not expose an X11 client window')
        self.window = int(window_id)

    def key(self, name, pressed):
        class KeyEvent(C.Structure):
            _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                ('display', C.c_void_p), ('window', C.c_ulong), ('root', C.c_ulong),
                ('subwindow', C.c_ulong), ('time', C.c_ulong), ('x', C.c_int), ('y', C.c_int),
                ('x_root', C.c_int), ('y_root', C.c_int), ('state', C.c_uint),
                ('keycode', C.c_uint), ('same_screen', C.c_int)]
        class Event(C.Union):
            _fields_ = [('key', KeyEvent), ('pad', C.c_long*24)]
        code = self.x.XKeysymToKeycode(self.display, self.x.XStringToKeysym(name.encode()))
        if not code:
            raise ValueError(name)
        event = Event()
        event.key = KeyEvent(type=2 if pressed else 3, send_event=True, display=self.display,
            window=self.window, root=self.x.XDefaultRootWindow(self.display), keycode=code, same_screen=True)
        if not self.x.XSendEvent(self.display, self.window, False, 1 if pressed else 2, C.byref(event)):
            raise RuntimeError('XSendEvent failed')
        self.x.XSync(self.display, False)

    def tap(self, name):
        self.key(name, True)
        # Let SDL consume the press before release, as with a physical key tap.
        time.sleep(.1)
        self.key(name, False)

    def close(self):
        self.x.XCloseDisplay(self.display)


def wait_for(controls, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        controls.poll()
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('Desktop keyboard event was not received')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = Stage2Config()
    report = dict(**provenance(cfg), passed=False, event_source='XSendEvent -> isolated SDL X11 window',
        human_tested=False, same_process_panda3d_pygame=False)
    # Events target an actual desktop window; no SDL dummy backend or global key injection.
    previous_driver = os.environ.get('SDL_VIDEODRIVER')
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    controls = keys = sim = logger = video = restore = None
    try:
        controls = KeyboardControls()
        keys = DesktopKeys(controls.desktop_info.get('window_id'))
        report['control_process_pid'] = controls.process.pid
        report['window_id'] = keys.window
        keys.tap('space');wait_for(controls, lambda: controls.paused)
        assert not controls.ready()
        keys.tap('n');wait_for(controls, lambda: controls.single_step)
        assert controls.ready() and not controls.ready()
        keys.tap('r');wait_for(controls, lambda: controls.reset_requested)
        keys.tap('m');wait_for(controls, lambda: controls.manual)
        keys.key('w', True);wait_for(controls, lambda: 'w' in controls.held)
        keys.key('a', True);wait_for(controls, lambda: 'a' in controls.held)
        assert controls.manual_action().tolist() == [1., 1.]
        keys.key('a', False);wait_for(controls, lambda: 'a' not in controls.held)
        keys.key('d', True);wait_for(controls, lambda: 'd' in controls.held)
        assert controls.manual_action()[0] == -1
        keys.key('d', False);wait_for(controls, lambda: 'd' not in controls.held)
        _, restore = configure_egl(cfg.egl_device)
        sim = make_env('S', 500000, cfg, camera=True)
        observation, _ = sim.reset(seed=500000)
        provider = NativeBEVProvider(sim, cfg.bev);provider.reset(observation)
        if controls.reset_requested:
            observation, _ = sim.reset(seed=500000)
            provider.reset(observation);controls.reset_requested = False
        keys.tap('space');wait_for(controls, lambda: not controls.paused)
        from evaluation.visuals import VideoWriter, compose
        video = VideoWriter('S', 500000, stage='stage2_keyboard_desktop', fps=10)
        logger = RerunLogger(spawn=False, save_path=args.output/'manual.rrd')
        for step in range(40):
            assert controls.ready()
            car = sim.agent
            state = np.array([*car.position, car.heading_theta, car.speed, *car.velocity], np.float32)
            action = controls.manual_action()
            bev = provider.capture().raster;rgb = third_person(sim)
            logger.set_step(step, step*cfg.dt_s);logger.log_bev(bev);logger.log_state(state)
            logger.log_camera(rgb);logger.log_telemetry(speed_m_s=car.speed,
                steering=action[0], throttle_brake=action[1])
            video.write(compose(bev, None, state, action, 'Desktop XSendEvent W -> isolated controls -> EGL', rgb))
            observation, _, tm, tc, _ = sim.step(action);provider.update(observation)
            if tm or tc:
                break
        report.update(simulation_steps=step+1, final_speed_m_s=float(sim.agent.speed), video=str(video.path))
        assert sim.agent.speed > 0
        keys.key('w', False);wait_for(controls, lambda: 'w' not in controls.held)
        keys.key('s', True);wait_for(controls, lambda: 's' in controls.held)
        assert controls.manual_action()[1] == -1
        keys.key('s', False);wait_for(controls, lambda: 's' not in controls.held)
        keys.tap('q');wait_for(controls, lambda: controls.quit)
        report.update(passed=True, verified_keys=['Space', 'N', 'R', 'M', 'W', 'A', 'S', 'D', 'Q'],
            paused_no_step=True, single_step_once=True, reset_applied=True)
    finally:
        if keys:
            # Release injected held keys before restoring the previous focus.
            for key in ('w', 'a', 's', 'd'):
                keys.key(key, False)
            keys.close()
        if controls: controls.close()
        if logger: logger.close()
        if video: video.close()
        if sim: sim.close()
        if restore: restore()
        if previous_driver is None: os.environ.pop('SDL_VIDEODRIVER', None)
        else: os.environ['SDL_VIDEODRIVER'] = previous_driver
        write_json(args.output/'report.json', report)
    print(report)


if __name__ == '__main__':
    main()
