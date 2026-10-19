"""Stage 2 native EGL environment; shares the accepted planar SemanticCamera."""
import hashlib
import json
import os

import numpy as np

from env.semantic_bev import SemanticBEVSpec, calibration, make_semantic_env, native_rgb, semantic_channels
from mini_wa_jepa.bev import BEVFrame, BEVProvider


class FullOffroadRule:
    """Only terminate after centre/corners all leave lanes for a grace interval.

    Native on_lane tests the centre alone. Repeated reward/cost/done queries in
    one decision must not count as multiple off-road simulation steps.
    """
    def __init__(self,dt_s,grace_s=1.):
        self.dt_s=dt_s;self.grace_s=grace_s;self.history={}

    def __call__(self,env,vehicle):
        step=env.engine.episode_step
        key=vehicle.id
        if vehicle.on_lane:
            self.history[key]=(step,0.)
            return False
        from mini_wa_jepa.geometry import box_corners
        from metadrive.utils.pg.utils import ray_localization
        points=box_corners(vehicle.position,vehicle.heading_theta,vehicle.LENGTH,vehicle.WIDTH)
        points=np.concatenate([np.asarray(vehicle.position)[None],points])
        entirely_off=not any(ray_localization(vehicle.heading,point,env.engine,
            use_heading_filter=False,return_on_lane=True)[1] for point in points)
        last,seconds=self.history.get(key,(step-1,0.))
        if step<last:seconds=0.;last=step-1
        if not entirely_off:seconds=0.
        elif step!=last:seconds+=self.dt_s
        self.history[key]=(step,seconds)
        return entirely_off and seconds+1e-8>=self.grace_s


def configure_egl(device=1):
    # The viewer is a separate process; leave the desktop environment available to it.
    from panda3d.core import loadPrcFileData
    loadPrcFileData('',f'load-display p3headlessgl\negl-device-index {device}\naudio-library-name null\nsync-video false')
    from env.render_compat import limit_filter_msaa
    return limit_filter_msaa(0)


def make_env(map_spec, seed_start, cfg, camera=False, num_scenarios=100, control_dt_s=None, environment_overrides=None):
    from metadrive.policy.env_input_policy import EnvInputPolicy
    spec=SemanticBEVSpec(cfg.bev.size,cfg.bev.extent_m,cfg.bev.height_m)
    dt_s=control_dt_s or cfg.dt_s
    repeat=max(1,round(dt_s/.02))
    physics_dt=dt_s/repeat
    extra=dict(map=map_spec,agent_policy=EnvInputPolicy,traffic_density=cfg.traffic_density,
        horizon=round(cfg.horizon*cfg.dt_s/dt_s),
        decision_repeat=repeat,physics_world_step_size=physics_dt,
        random_spawn_lane_index=False,
        agent_configs={'default_agent':{'spawn_lane_index':('>','>>',1)}},
        # Markings and brief centre-off-lane excursions remain recoverable.
        on_continuous_line_done=False,on_broken_line_done=False,out_of_road_done=True,
        show_interface=False,interface_panel=[],camera_smooth=False)
    if camera:
        from metadrive.component.sensors.semantic_camera import SemanticCamera
        extra.update(sensors={'bev_semantic':(SemanticCamera,128,128),'main_camera':()},
                     window_size=(640,360))
    if environment_overrides is not None:extra.update(environment_overrides)
    return make_semantic_env(spec,num_scenarios,seed_start,cfg.horizon,overrides=extra,
                             road_termination=FullOffroadRule(dt_s,cfg.offroad_grace_s))


class NativeBEVProvider(BEVProvider):
    def __init__(self,sim,cfg):
        self.sim=sim
        self.spec=SemanticBEVSpec(cfg.size,cfg.extent_m,cfg.height_m)
        self.bev=None

    def reset(self,observation=None):
        features=self.sim.current_map.get_map_features(1.0)
        serial=[]
        for key,item in sorted(features.items()):
            serial.append([key,item['type'],np.round(np.asarray(item.get('polygon',item.get('polyline',[]))),4).tolist()])
        self.geometry_hash=hashlib.sha256(json.dumps(serial,separators=(',',':')).encode()).hexdigest()
        self.calibration=calibration(self.sim,self.spec)
        if max(row['max_error_pixel'] for row in self.calibration.values())>=.05:
            raise RuntimeError(f'BEV calibration failure: {self.calibration}')
        if observation is None:
            sensor=self.sim.engine.get_sensor('bev_semantic')
            rgb=np.ascontiguousarray(sensor.perceive(to_float=False)[...,::-1])
            self.bev=semantic_channels(rgb,self.sim.agent.LENGTH,self.sim.agent.WIDTH,self.spec)
        else:self.update(observation)

    def update(self,observation):
        self.bev=semantic_channels(native_rgb(observation),self.sim.agent.LENGTH,self.sim.agent.WIDTH,self.spec)

    def capture(self):
        if self.bev is None:raise RuntimeError('Reset provider before capture')
        return BEVFrame(self.bev,dict(channels=['drivable','boundary_static','other_actors','visibility'],
            extent_m=self.spec.extent_m,metres_per_pixel=self.spec.extent_m/self.spec.size,
            ego_pixel=[(self.spec.size-1)/2]*2,axes='x_forward_y_left_heading_ccw',
            orientation='ego_heading_up',layout='CHW',value_range=[0,255],
            visibility='full_ground_truth_no_degradation',renderer='native_semantic_camera_egl'))


def third_person(sim):
    return np.ascontiguousarray(sim.main_camera.perceive(to_float=False)[...,::-1])
