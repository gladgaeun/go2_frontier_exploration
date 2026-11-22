# Unit tests for the spatial value map (value_map.py, TravExplorer Sec. V-B, eq. 7).
#   python3 -m pytest trav_exploration/test/test_value_map.py

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.value_map import ValueMap, utilities, view_weight  # noqa: E402

FOV = math.radians(79.0)


def test_view_weight_centre_edge_outside():
    w = view_weight([0.0, FOV / 4, FOV / 2, FOV], FOV)
    assert abs(w[0] - 1.0) < 1e-12
    assert abs(w[1] - 0.5) < 1e-12                       # cos^2(pi / 4)
    assert w[2] < 1e-12 and w[3] == 0.0                  # boundary, outside


def test_first_observation_sets_value_and_reliability():
    v = ValueMap()
    v.update([5, 3], [0.8, 0.4], 0.9)
    s, r = v.get([3, 5, 7])
    assert np.allclose(s, [0.9, 0.9, 0.0]) and np.allclose(r, [0.4, 0.8, 0.0])
    assert list(v.keys) == [3, 5]                        # sorted


def test_eq7_fusion():
    v = ValueMap()
    v.update([1], [0.5], 0.2)
    v.update([1], [1.0], 0.8)
    s, r = v.get([1])
    assert abs(s[0] - (0.5 * 0.2 + 1.0 * 0.8) / 1.5) < 1e-12
    assert abs(r[0] - (0.25 + 1.0) / 1.5) < 1e-12


def test_reliable_value_resists_a_glancing_outlier():
    # seen head-on three times with a low score, then once near the FOV edge with a high one
    v = ValueMap()
    for _ in range(3):
        v.update([1], [1.0], 0.1)
    w_edge = float(view_weight(0.45 * FOV, FOV))
    v.update([1], [w_edge], 0.9)
    s, _ = v.get([1])
    assert s[0] < 0.13                                   # (0.1 + 0.025 * 0.9) / 1.025 = 0.12
    head_on = ValueMap()
    for _ in range(3):
        head_on.update([1], [1.0], 0.1)
    head_on.update([1], [1.0], 0.9)
    assert abs(head_on.get([1])[0][0] - 0.5) < 1e-12    # the same score seen head-on counts fully


def test_pixels_of_one_voxel_use_their_mean_weight_and_new_keys_merge():
    v = ValueMap()
    v.update([2, 2, 9], [0.2, 0.6, 1.0], 0.5)
    _, r = v.get([2])
    assert abs(r[0] - 0.4) < 1e-12
    v.update([4, 9, 1], [1.0, 1.0, 1.0], 0.5)
    assert list(v.keys) == [1, 2, 4, 9] and len(v) == 4
    v.update([7], [0.0], 0.5)                            # zero weight: not observed
    assert 7 not in v.keys


def test_utilities_max_in_neighbourhood_same_floor():
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0], [0.2, 0.0, 3.0]])
    vals = np.array([0.1, 0.7, 0.9, 1.0])
    u = utilities(pts, vals, [(0.0, 0.0, 0.0), (5.5, 0.0, 0.0), (0.0, 0.0, 3.0)], radius=1.0, dz=0.3)
    assert np.allclose(u, [0.7, 0.0, 1.0])               # upper-floor voxel not counted below


def test_rviz_rainbow_ends_and_middle():
    from trav_exploration.value_map import rviz_rainbow
    c = rviz_rainbow([0.0, 0.4, 0.6, 1.0])
    assert np.allclose(c[0], [1, 0, 0]) and np.allclose(c[3], [1, 0, 1])     # red low, magenta high
    assert np.allclose(c[1], [0, 1, 0]) and np.allclose(c[2], [0, 1, 1])     # green, cyan


def test_score_overlay_tints_the_centre_most():
    from trav_exploration.value_map import score_overlay
    rgb = np.zeros((4, 101, 3), dtype=np.uint8)
    fx = 50.5 / math.tan(FOV / 2)
    img, w = score_overlay(rgb, 0.0, fx, 50.5, FOV)
    assert abs(w[50] - 1.0) < 1e-3 and w[0] < 1e-3                            # centre / edge column
    assert img[0, 50, 0] > 100 and img[0, 0, 0] < 2 and img[0, 50, 1] == 0    # red, centre only
