"""Navigation adapter around the independent trajectory tracker.

No imported MetaDrive driving policy. Geometry, vehicle state and object states
are teacher-only data; the learner receives only the provider contract.
"""
import numpy as np

from configs.stage2 import ActionNoiseConfig, ExpertConfig
from controller.pure_pursuit import SpeedPID, pursuit_steering
from mini_wa_jepa.geometry import world_to_ego, wrap_angle


class NavigationExpert:
    def __init__(self, sim, config=ExpertConfig(), dt=0.1):
        self.sim, self.cfg = sim, config
        self.pid = SpeedPID(config.speed_kp, config.speed_ki, config.speed_kd, dt)
        car = sim.agent
        self.wheelbase = float(car.FRONT_WHEELBASE + car.REAR_WHEELBASE)
        self.max_steer = np.deg2rad(car.max_steering)

    def route(self, lookahead):
        car, nav = self.sim.agent, self.sim.agent.navigation
        # Keep a stable lane index; choose the closest reference segment to
        # handle the navigation module's delayed checkpoint transition.
        candidates = list(nav.current_ref_lanes) + list(nav.next_ref_lanes or [])
        lane_id = 1
        choices = [l for l in candidates if l.index[-1] == min(lane_id, len(nav.current_ref_lanes)-1)]
        lane = min(choices or candidates, key=lambda l: l.distance(car.position))
        s, lat = lane.local_coordinates(car.position)
        points = []
        remaining = lookahead
        cursor = max(0.0, float(s))
        current = lane
        # Traverse actual connected route lanes using checkpoints, never a
        # straight extrapolation beyond the end of a curved lane.
        for _ in range(8):
            available = max(0.0, current.length - cursor)
            if remaining <= available:
                target_s = cursor + remaining
                return current.position(target_s, 0), lane, float(lat), current.heading_theta_at(target_s)
            remaining -= available
            end = current.index[1]
            checkpoints = nav.checkpoints
            if end not in checkpoints or checkpoints.index(end) >= len(checkpoints)-1:
                return current.position(current.length, 0), lane, float(lat), current.heading_theta_at(current.length)
            idx = checkpoints.index(end)
            next_lanes = self.sim.current_map.road_network.graph[end][checkpoints[idx+1]]
            current = next_lanes[min(lane_id, len(next_lanes)-1)]
            cursor = 0.0
        raise RuntimeError("Lookahead traversed too many route segments")

    def act(self):
        car, cfg = self.sim.agent, self.cfg
        lookahead = max(cfg.min_lookahead_m, cfg.lookahead_time_s * car.speed)
        target, lane, lateral, heading_ahead = self.route(lookahead)
        target_local = world_to_ego(target, car.position, car.heading_theta)
        steer = pursuit_steering(target_local, self.wheelbase, self.max_steer)
        s, _ = lane.local_coordinates(car.position)
        curvature = abs(float(wrap_angle(heading_ahead - lane.heading_theta_at(np.clip(s, 0, lane.length))))) / max(lookahead, 1)
        speed_target = min(cfg.target_speed_m_s, np.sqrt(cfg.max_lateral_accel_m_s2 / max(curvature, 1e-4)))
        # Simple headway cap along the reference lane, including connected
        # following lanes. Does not implement lane changes or replace the PID.
        for obj in self.sim.engine.get_objects(lambda x: getattr(x, "metadrive_type", None) == "VEHICLE").values():
            if obj is car:
                continue
            relative = world_to_ego(obj.position, car.position, car.heading_theta)
            if relative[0] <= 0 or np.linalg.norm(relative) > 45:
                continue
            os, ol = lane.local_coordinates(obj.position)
            if abs(ol) < lane.width_at(np.clip(os, 0, lane.length)) * 0.5 + obj.WIDTH * 0.5:
                gap = os - s - (car.LENGTH + obj.LENGTH) / 2
                if gap > -1 and gap < 40:
                    front_speed = max(0.0, float(np.dot(obj.velocity, lane.heading_at(np.clip(os, 0, lane.length)))))
                    cap = max(0.0, front_speed + (gap - cfg.standstill_gap_m - cfg.headway_s * car.speed) / cfg.headway_s)
                    speed_target = min(speed_target, cap)
        throttle = self.pid.step(speed_target, car.speed)
        return np.array([steer, throttle], dtype=np.float32), dict(
            lateral_m=lateral, target_speed_m_s=float(speed_target),
            target_ego=target_local.tolist(), curvature=float(curvature))


class ActionNoise:
    def __init__(self, seed, config=ActionNoiseConfig()):
        self.rng, self.cfg = np.random.default_rng(seed), config
        self.remaining = 0
        self.recovery_remaining = 0
        self.offset = np.zeros(2, np.float32)

    def apply(self, action):
        if not self.cfg.enabled:
            return action.copy(), False
        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            return action.copy(), False
        if self.remaining == 0 and self.rng.random() < self.cfg.event_probability:
            self.remaining = int(self.rng.integers(self.cfg.duration_min, self.cfg.duration_max+1))
            self.offset = self.rng.normal(0, [self.cfg.steering_std, self.cfg.throttle_std]).astype(np.float32)
        active = self.remaining > 0
        if active:
            self.remaining -= 1
            if self.remaining == 0:
                self.recovery_remaining = self.cfg.recovery_steps
        return np.clip(action + self.offset if active else action, -1, 1).astype(np.float32), active
