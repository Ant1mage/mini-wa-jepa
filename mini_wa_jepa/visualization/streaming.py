"""Adapters for recorded frames, independent of the simulator and Rerun imports."""
from pathlib import Path

import numpy as np


def add_rerun_arguments(parser):
    parser.add_argument('--rerun', action='store_true', help='Stream to an independent Rerun viewer')
    parser.add_argument('--rerun-save', type=Path, metavar='FILE.rrd', help='Save a Rerun recording (no viewer required)')


def make_rerun_logger(args, app_id):
    spawn = args.rerun or args.live
    if not spawn and args.rerun_save is None:
        return None
    from mini_wa_jepa.visualization import RerunLogger
    return RerunLogger(app_id=app_id, spawn=spawn, save_path=args.rerun_save)


def log_recorded_frame(logger, row, *, dt_s, timeline_step, extent_m=64.0,
                       planned=None, ground_truth=None):
    """Keep BEV, pre-action state, controls and future labels on one timestamp."""
    logger.set_step(timeline_step, sim_time_s=timeline_step * dt_s)
    logger.log_context(episode_id=int(row['episode_id']), episode_step=int(row['step_id']),
                       seed=int(row['seed']), noise_active=bool(row['noise_active']),
                       manual=bool(row.get('manual', False)))
    logger.log_bev(row['bev'], extent_m=extent_m)
    logger.log_state(row['state'])
    logger.log_camera(row.get('camera_rgb'))
    logger.log_telemetry(speed_m_s=row['state'][3], steering=row['executed_action'][0],
                         throttle_brake=row['executed_action'][1],
                         lateral_error_m=row.get('lateral_m'))
    if ground_truth is None and 'future_trajectory' in row:
        ground_truth = row['future_trajectory'][row['future_valid']]
    empty = np.empty((0, 2))
    logger.log_trajectories(planned=planned if planned is not None else empty,
                            ground_truth=ground_truth if ground_truth is not None else empty,
                            size=row['bev'].shape[-1], extent_m=extent_m)
