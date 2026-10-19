"""Independently implemented pure pursuit steering and anti-windup speed PID."""
import numpy as np


class SpeedPID:
    def __init__(self, kp=0.5, ki=0.08, kd=0.02, dt=0.1):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral, self.previous = 0.0, None

    def step(self, target, speed):
        error = float(target - speed)
        derivative = 0 if self.previous is None else (error - self.previous) / self.dt
        candidate = np.clip(self.integral + error * self.dt, -3.0, 3.0)
        raw = self.kp * error + self.ki * candidate + self.kd * derivative
        if abs(raw) < 1 or np.sign(raw) != np.sign(error):
            self.integral = float(candidate)
        self.previous = error
        return float(np.clip(raw, -1, 1))


def pursuit_steering(target_ego, wheelbase, max_steering_rad):
    x, y = target_ego
    radius_squared = max(x*x + y*y, 1e-4)
    angle = np.arctan2(2 * wheelbase * y, radius_squared)
    return float(np.clip(angle / max_steering_rad, -1, 1))
