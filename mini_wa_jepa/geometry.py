"""Shared convention: x forward, y left, heading CCW, SI units."""
import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def world_to_ego(points, position, heading):
    d = np.asarray(points) - np.asarray(position)
    c, s = np.cos(heading), np.sin(heading)
    return d @ np.array([[c, -s], [s, c]])


def ego_to_world(points, position, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(points) @ np.array([[c, s], [-s, c]]) + np.asarray(position)


def ego_to_pixels(points, size=64, extent_m=64.0):
    points = np.asarray(points)
    return np.stack(((size - 1) / 2 - points[..., 1] * size / extent_m,
                     (size - 1) / 2 - points[..., 0] * size / extent_m), axis=-1)


def box_corners(position, heading, length, width):
    local = np.array([[length/2, width/2], [length/2, -width/2],
                      [-length/2, -width/2], [-length/2, width/2]])
    return ego_to_world(local, position, heading)
