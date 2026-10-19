"""Thin wrappers around APIs checked against the pinned MetaDrive source.

The semantic image here is a native visualization, NOT the stage 2 binary BEV.
RGB is only an installation diagnostic and is not a model input.
"""
import numpy as np

from configs.stage1 import Stage1Config


def make_env(cfg: Stage1Config):
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from metadrive.constants import DEFAULT_SENSOR_HPR, DEFAULT_SENSOR_OFFSET

    class VerificationEnv(MetaDriveEnv):
        def reset_sensors(self):
            super().reset_sensors()
            # Make ego attachment explicit via the documented developer hook.
            # MetaDrive 0.4.3 also creates an internal main camera when its
            # image is not registered, and the native hook already mounts RGB.
            sensor = self.engine.get_sensor("rgb_camera")
            sensor.track(self.agent.origin, DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
            self.engine.taskMgr.step()
            self.engine.taskMgr.step()

    return VerificationEnv(dict(
        use_render=False, image_observation=True, image_on_cuda=False,
        sensors={"rgb_camera": (RGBCamera, cfg.rgb_width, cfg.rgb_height)},
        vehicle_config={"image_source": "rgb_camera", "show_navi_mark": False},
        agent_policy=IDMPolicy,  # built-in driver only for installation testing
        num_scenarios=cfg.episodes, start_seed=cfg.start_seed,
        map=cfg.map, store_map=False, horizon=cfg.horizon,
        traffic_density=cfg.traffic_density, stack_size=1, norm_pixel=False,
        show_terrain=False, interface_panel=[], show_interface=False,
        log_level=50,
    ))


def capture_bev(sim, cfg: Stage1Config):
    image = sim.render(
        mode="topdown", window=False, semantic_map=True,
        target_agent_heading_up=True, center_on_map=False,
        screen_size=(cfg.bev_size, cfg.bev_size),
        film_size=(cfg.film_size, cfg.film_size),
        scaling=cfg.bev_pixels_per_metre, num_stack=1,
        draw_contour=False,
    )
    return np.ascontiguousarray(image)


def capture_rgb(sim):
    # BaseCamera.get_image explicitly supports RGB; observation arrays use BGR.
    return np.ascontiguousarray(
        sim.engine.get_sensor("rgb_camera").get_image(sim.agent, mode="rgb")
    )


def state_snapshot(sim):
    car = sim.agent
    nav = car.navigation
    lane = nav.current_ref_lanes[0]
    longitudinal, lateral = lane.local_coordinates(car.position)
    target_s = float(np.clip(longitudinal + 10.0, 0, lane.length))
    return dict(
        position_m=np.asarray(car.position).tolist(),
        heading_rad=float(car.heading_theta), speed_m_s=float(car.speed),
        velocity_m_s=np.asarray(car.velocity).tolist(),
        current_lane_index=list(car.lane_index),
        reference_lanes=len(nav.current_ref_lanes),
        next_reference_lanes=len(nav.next_ref_lanes or []),
        checkpoints=list(nav.checkpoints),
        lane_length_m=float(lane.length),
        lane_coordinates_m=[float(longitudinal), float(lateral)],
        reference_point_10m=np.asarray(lane.position(target_s, 0)).tolist(),
        reference_heading_rad=float(lane.heading_theta_at(target_s)),
        destination_m=np.asarray(nav.final_lane.position(nav.final_lane.length, 0)).tolist(),
        route_completion=float(nav.route_completion),
        dynamics=car.get_dynamics_parameters(),
        dimensions_m=[float(car.LENGTH), float(car.WIDTH), float(car.HEIGHT)],
    )
