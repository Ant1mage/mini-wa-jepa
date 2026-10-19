"""Native map geometry + object boxes, rasterized in the shared ego frame."""
import hashlib
import json

import cv2
import numpy as np

from configs.stage2 import BEVConfig
from mini_wa_jepa.bev import BEVFrame, BEVProvider
from mini_wa_jepa.geometry import box_corners, ego_to_pixels, world_to_ego


class SimulatorGroundTruthBEV(BEVProvider):
    def __init__(self, sim, config=BEVConfig()):
        if config.size != 64 or config.extent_m <= 0:
            raise ValueError("Stage 2 contract is 64x64 with positive extent")
        self.sim, self.config = sim, config
        self.features = []

    def reset(self):
        from metadrive.type import MetaDriveType as T
        native = self.sim.current_map.get_map_features(1.0)
        self.features = []
        digest = []
        for key, f in sorted(native.items()):
            typ = f["type"]
            if T.is_lane(typ):
                points, channel, fill = f.get("polygon"), 0, True
            elif T.is_road_boundary_line(typ) or T.is_solid_line(typ):
                points, channel, fill = f.get("polyline"), 1, False
            elif T.is_sidewalk(typ):
                points, channel, fill = f.get("polygon"), 1, True
            else:
                continue
            if points is None or len(points) < 2:
                continue
            points = np.asarray(points, dtype=np.float64)[..., :2]
            self.features.append((points, channel, fill, points.min(0), points.max(0)))
            digest.append([key, typ, np.round(points, 4).tolist()])
        self.geometry_hash = hashlib.sha256(json.dumps(digest, separators=(",", ":")).encode()).hexdigest()

    def capture(self):
        from metadrive.type import MetaDriveType as T
        cfg, car = self.config, self.sim.agent
        bev = np.zeros((3, cfg.size, cfg.size), dtype=np.uint8)
        position, heading = np.asarray(car.position), car.heading_theta
        radius = cfg.extent_m / np.sqrt(2) + 3

        def draw(points, channel, fill):
            pixels = ego_to_pixels(world_to_ego(points, position, heading), cfg.size, cfg.extent_m)
            fixed = np.rint(np.clip(pixels, -1e5, 1e5) * 256).astype(np.int32)
            if fill:
                cv2.fillPoly(bev[channel], [fixed], 255, lineType=cv2.LINE_8, shift=8)
            else:
                width = max(1, round(cfg.line_width_m * cfg.size / cfg.extent_m))
                cv2.polylines(bev[channel], [fixed], False, 255, thickness=width, lineType=cv2.LINE_8, shift=8)

        for points, channel, fill, lo, hi in self.features:
            if np.all(hi >= position - radius) and np.all(lo <= position + radius):
                draw(points, channel, fill)
        objects = self.sim.engine.get_objects(lambda obj:
            T.is_vehicle(getattr(obj, "metadrive_type", None)) or
            T.is_traffic_object(getattr(obj, "metadrive_type", None)))
        for obj in objects.values():
            if obj is car or np.linalg.norm(np.asarray(obj.position) - position) > radius:
                continue
            draw(box_corners(obj.position, obj.heading_theta, obj.LENGTH, obj.WIDTH),
                 2 if T.is_vehicle(obj.metadrive_type) else 1, True)
        return BEVFrame(bev, dict(channels=["drivable", "boundary_obstacle", "other_vehicle"],
            extent_m=cfg.extent_m, metres_per_pixel=cfg.extent_m / cfg.size,
            ego_pixel=[(cfg.size-1)/2]*2, axes="x_forward_y_left_heading_ccw",
            orientation="ego_heading_up", layout="CHW", value_range=[0,255]))
