"""Stage 1 v2: native orthographic semantic camera and explicit CHW mapping.

This is the new environment diagnostic only. It does not modify the legacy
64px dataset/provider or implement the later visibility degrader.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SemanticBEVSpec:
    size: int = 128
    extent_m: float = 64.0
    height_m: float = 50.0


def make_semantic_env(spec=SemanticBEVSpec(), episodes=100, start_seed=0, horizon=500, overrides=None, road_termination=None):
    from metadrive.component.sensors.semantic_camera import SemanticCamera
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from panda3d.core import OrthographicLens

    class SemanticEnv(MetaDriveEnv):
        def _is_out_of_road(self,vehicle):
            if road_termination is None:return super()._is_out_of_road(vehicle)
            return road_termination(self,vehicle)

        def reset_sensors(self):
            super().reset_sensors()
            sensor = self.engine.get_sensor('bev_semantic')
            lens = OrthographicLens()
            lens.setFilmSize(spec.extent_m, spec.extent_m)
            lens.setNearFar(0.1, spec.height_m + 100)
            sensor.cam.node().setLens(lens)
            # World-mounted planar pose avoids inheriting chassis pitch/roll.
            sensor.cam.reparentTo(self.engine.render)
            self.engine.taskMgr.remove('update_planar_semantic_bev')
            def update(task):
                car=self.agent
                sensor.cam.setPos(float(car.position[0]),float(car.position[1]),spec.height_m)
                sensor.cam.setHpr(float(np.rad2deg(car.heading_theta)-90),-90,0)
                return task.cont
            self.engine.taskMgr.add(update,'update_planar_semantic_bev',sort=-10)
            self.engine.taskMgr.step()
            self.engine.taskMgr.step()

        def close(self):
            if self.engine is not None:
                self.engine.taskMgr.remove('update_planar_semantic_bev')
            super().close()

    config=dict(use_render=False, image_observation=True, image_on_cuda=False,
        sensors={'bev_semantic': (SemanticCamera, spec.size, spec.size)},
        vehicle_config={'image_source':'bev_semantic','show_navi_mark':False},
        stack_size=1, norm_pixel=False, agent_policy=IDMPolicy,
        num_scenarios=episodes,start_seed=start_seed,map=3,horizon=horizon,store_map=False,
        traffic_density=0.1,show_terrain=True,show_interface=False,interface_panel=[],
        multi_thread_render=False,log_level=50)
    config.update(overrides or {})
    return SemanticEnv(config)


def native_rgb(observation):
    """The sensor observation is BGR, HWC x stack, already read back by step."""
    return np.ascontiguousarray(observation['image'][..., -1][..., ::-1], dtype=np.uint8)


def semantic_channels(rgb, ego_length, ego_width, spec=SemanticBEVSpec()):
    from metadrive.constants import Semantics

    def mask(*categories):
        result = np.zeros(rgb.shape[:2], bool)
        for category in categories:
            r,g,b = category.color
            result |= packed == ((r<<16)|(g<<8)|b)
        return result

    packed = (rgb[...,0].astype(np.uint32)<<16) | (rgb[...,1].astype(np.uint32)<<8) | rgb[...,2]
    bev = np.zeros((4,spec.size,spec.size), np.uint8)
    bev[0] = mask(Semantics.ROAD, Semantics.CROSSWALK, Semantics.LANE_LINE) * 255
    bev[1] = mask(Semantics.SIDEWALK, Semantics.FENCE, Semantics.TRAFFIC_SIGN) * 255
    bev[2] = mask(Semantics.CAR, Semantics.TRUCK, Semantics.BUS, Semantics.BIKE, Semantics.PEDESTRIAN) * 255
    # Remove ego using its known metric footprint; all other actors remain.
    center = (spec.size - 1) / 2
    half_x = ego_width * spec.size/spec.extent_m / 2
    half_y = ego_length * spec.size/spec.extent_m / 2
    ego_mask = np.zeros((spec.size,spec.size), np.uint8)
    points = np.rint(np.array([[center-half_x,center-half_y],[center+half_x,center-half_y],
        [center+half_x,center+half_y],[center-half_x,center+half_y]])*256).astype(np.int32)
    cv2.fillPoly(ego_mask,[points],255,shift=8)
    bev[2,ego_mask>0] = 0
    # No driver-eye occlusion is simulated in perfect-GT stage 1.
    bev[3] = 255
    return bev


def calibration(sim, spec=SemanticBEVSpec()):
    from panda3d.core import Point2, Point3
    from mini_wa_jepa.geometry import ego_to_world
    cam = sim.engine.get_sensor('bev_semantic').cam
    result = {}
    for name, local in [('center',(0,0)),('forward_10m',(10,0)),('left_10m',(0,10))]:
        world = ego_to_world(local,sim.agent.position,sim.agent.heading_theta)
        point = cam.getRelativePoint(sim.engine.render,Point3(float(world[0]),float(world[1]),0))
        projected = Point2()
        assert cam.node().getLens().project(point,projected)
        pixels = [(float(projected.x)+1)*spec.size/2-0.5,
                  (1-float(projected.y))*spec.size/2-0.5]
        expected = [(spec.size-1)/2-local[1]*spec.size/spec.extent_m,
                    (spec.size-1)/2-local[0]*spec.size/spec.extent_m]
        result[name] = dict(actual_pixel=pixels,expected_pixel=expected,
                           max_error_pixel=float(np.max(np.abs(np.array(pixels)-expected))))
    return result
