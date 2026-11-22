"""First z_r estimate of voxel_mapper_node: the lowest floor cluster near camera_height below the camera."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.voxel_mapper_node import VoxelMapperNode  # noqa: E402


def _node():
    return SimpleNamespace(support_z=None, camera_height=0.88, start_floor_band=0.10, floor_radius=3.0,
                           floor_max_above=0.1, floor_max_jump=0.15, floor_min_points=50)


def _plane(z, n, x0=1.0):
    rng = np.random.default_rng(0)
    xy = rng.uniform([x0, -1.0], [x0 + 1.5, 1.0], size=(n, 2))
    return np.column_stack([xy, np.full(n, z) + rng.normal(0.0, 0.004, n)])


def test_first_estimate_is_the_lowest_cluster_not_the_largest():
    # floor 0.177 m under base_link (z 0), a low furniture top 0.25 m over the floor with more
    # points (00880 0093: the old mode took the top, z_r +0.075)
    floor, top = -0.177, 0.075
    pts = np.vstack([_plane(floor, 120), _plane(top, 600, x0=0.5)])
    cam_z = floor + 0.88
    est = VoxelMapperNode._estimate_floor(_node(), pts, np.zeros(3), cam_z=cam_z)
    assert abs(est - floor) < 0.02


def test_first_estimate_leaves_out_a_lower_storey():
    # the storey below seen down a stairwell is the lowest cluster, but far outside the band
    floor = -0.08
    pts = np.vstack([_plane(floor, 200), _plane(floor - 2.9, 2000)])
    est = VoxelMapperNode._estimate_floor(_node(), pts, np.zeros(3), cam_z=floor + 0.88)
    assert abs(est - floor) < 0.02


def test_first_estimate_none_without_enough_floor_points():
    pts = _plane(-0.08, 20)
    assert VoxelMapperNode._estimate_floor(_node(), pts, np.zeros(3), cam_z=0.80) is None
