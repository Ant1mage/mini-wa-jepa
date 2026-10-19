"""Causal candidate commands and coordinate transforms for real branch rollouts."""
import numpy as np
from mini_wa_jepa.geometry import ego_to_world, world_to_ego, wrap_angle


def make_candidates(expert, amplitude=1.5):
    """Construct commands before execution; never use realized future states."""
    car = expert.sim.agent
    speed = float(np.clip(car.speed, 3., 8.))
    times = np.arange(1, 11) * .2
    result = []
    for name, offset, factor in [('keep', 0., 1.), ('left', amplitude, 1.),
                                  ('right', -amplitude, 1.), ('slow', 0., .55)]:
        distance = speed * times * factor
        rows = []
        for t, s in zip(times, distance):
            point, _, _, angle = expert.route(float(s))
            # Smooth lateral displacement in the reference lane's left direction.
            q = t / 2.
            lateral = offset * (3*q*q - 2*q*q*q)
            slope = offset * (3*q - 3*q*q) / max(speed*factor, 1.)
            point = np.asarray(point) + lateral*np.array([-np.sin(angle), np.cos(angle)])
            xy = world_to_ego(point, car.position, car.heading_theta)
            rows.append([*xy, speed*factor, float(wrap_angle(angle + np.arctan(slope)-car.heading_theta))])
        result.append(np.asarray(rows, np.float32))
    return np.stack(result)


def remaining_command(command, anchor_state, current_state, elapsed_s):
    """Re-express a fixed world command for the existing ego-frame tracker."""
    command = np.asarray(command)
    world_xy = ego_to_world(command[:, :2], anchor_state[:2], anchor_state[2])
    times = np.arange(1, 11)*.2
    query = np.minimum(elapsed_s + times, 2.)
    extended_t = np.r_[0., times]
    world_xy = np.vstack([anchor_state[:2], world_xy])
    xy = np.stack([np.interp(query, extended_t, world_xy[:, j]) for j in range(2)], -1)
    speed = np.interp(query, times, command[:, 2])
    angle = np.interp(query, times, np.unwrap(command[:, 3]+anchor_state[2]))
    return np.column_stack([world_to_ego(xy, current_state[:2], current_state[2]),
                            speed, wrap_angle(angle-current_state[2])]).astype(np.float32)


def realized_waypoints(states, anchor_index, indices):
    states = np.asarray(states)
    anchor = states[anchor_index]
    future = states[np.asarray(indices)]
    return np.column_stack([world_to_ego(future[:, :2], anchor[:2], anchor[2]),
                            future[:, 3], wrap_angle(future[:, 2]-anchor[2])]).astype(np.float32)
