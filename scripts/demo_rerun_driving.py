"""Demonstration script showing complete driving telemetry (steering, throttle, trajectories) in Rerun."""
import argparse
from pathlib import Path
import numpy as np

from mini_wa_jepa.visualization.rerun_logger import RerunLogger


def generate_driving_demo(num_steps: int = 120, save_path: str = "artifacts/restart_stage1/driving_telemetry_demo.rrd"):
    """Generate a realistic driving episode with steering, throttle, speed, and trajectories."""
    logger = RerunLogger(app_id="mini-wa-jepa-driving-demo", spawn=False, save_path=save_path)

    # Base BEV map template: straight road that curves right
    h, w = 128, 128
    cx, cy = w // 2, h // 2

    # Simulated driving states
    speed = 0.0
    steer = 0.0
    throttle = 0.5
    lateral_error = 0.0

    print(f"Generating {num_steps} steps of driving telemetry demo with steering, throttle, and trajectories...")

    for step in range(num_steps):
        dt = 0.1
        sim_time = step * dt

        # Simulate vehicle physics and driving events:
        # Phase 1 (0..30): Accelerating straight
        if step < 30:
            throttle = 0.6
            steer = 0.0
            target_steer = 0.0
            lateral_error = 0.05 * np.sin(step * 0.2)
        # Phase 2 (30..60): Right turn curve
        elif step < 60:
            throttle = 0.3
            target_steer = 0.35 + 0.05 * np.sin(step * 0.3)
            steer += (target_steer - steer) * 0.2
            lateral_error = 0.15 * np.sin(step * 0.1)
        # Phase 3 (60..80): 10% disturbance pulse (sudden left drift) & expert recovery
        elif step < 80:
            if step < 68:
                # Disturbance pulse!
                throttle = 0.4
                steer = -0.45
                lateral_error += 0.08
            else:
                # Expert takes over and countersteers to recover
                throttle = 0.2
                steer = 0.40
                lateral_error -= 0.06
        # Phase 4 (80..120): Braking and straight cruise
        else:
            throttle = -0.2 if speed > 6.0 else 0.25
            steer = 0.0
            lateral_error = 0.02 * np.sin(step * 0.1)

        # Speed update
        speed = max(0.0, speed + (throttle * 3.0 - 0.2) * dt)

        # Generate realistic 4-channel BEV
        bev = np.zeros((4, h, w), dtype=np.uint8)
        # Road lane polygon (curving slightly depending on steer)
        road_width = 36
        offset = int(steer * 20.0)
        bev[0, :, max(0, cx - road_width // 2 - offset):min(w, cx + road_width // 2 - offset)] = 255
        # Road boundaries
        bev[1, :, max(0, cx - road_width // 2 - offset - 2):max(0, cx - road_width // 2 - offset)] = 255
        bev[1, :, min(w, cx + road_width // 2 - offset):min(w, cx + road_width // 2 - offset + 2)] = 255
        # Dynamic car ahead
        ahead_y = int(cy - 25 - (speed * 0.5))
        ahead_x = cx - offset
        if 0 <= ahead_y < h:
            bev[2, max(0, ahead_y - 6):min(h, ahead_y + 6), max(0, ahead_x - 3):min(w, ahead_x + 3)] = 255
        # Full visibility
        bev[3, :, :] = 255

        # Future 10 trajectory points (0.2s spacing, 2.0s horizon)
        traj_planned = []
        traj_gt = []
        for k in range(1, 11):
            t = k * 0.2
            dx = speed * t
            # Lateral curve extrapolation
            dy_planned = -0.5 * steer * (dx ** 1.3)
            dy_gt = -0.5 * steer * (dx ** 1.3) + (0.1 if step >= 60 and step < 68 else 0.0)
            traj_planned.append([dx, dy_planned])
            traj_gt.append([dx, dy_gt])

        traj_planned = np.asarray(traj_planned, dtype=np.float32)
        traj_gt = np.asarray(traj_gt, dtype=np.float32)

        # Push to Rerun timeline
        logger.set_step(step=step, sim_time_s=sim_time)
        logger.log_bev(bev)
        logger.log_telemetry(
            speed_m_s=speed,
            steering=steer,
            throttle_brake=throttle,
            lateral_error_m=lateral_error,
            route_completion=float(step) / num_steps,
        )
        logger.log_trajectories(planned=traj_planned, ground_truth=traj_gt)

    print(f"Saved complete telemetry demo to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--save", type=str, default="artifacts/restart_stage1/driving_telemetry_demo.rrd")
    args = parser.parse_args()
    generate_driving_demo(args.steps, args.save)
