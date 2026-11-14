# Unit tests for goal_selection (ported 2D goal ranking and placement), no ROS.
#   python3 -m pytest trav_exploration/test/test_goal_selection.py

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.goal_selection import pull_back, select_frontier, snap_to_clear_cell  # noqa: E402


def test_ranking_like_2d():
    clusters = [(3.0, 0.0, 0.0, 10), (1.0, 0.0, 0.0, 10), (5.0, 0.0, 0.0, 40)]
    # score = 0.5 / d + 0.5 * size: size dominates, then the closer one
    assert select_frontier(clusters, 0, 0.0, 0.0)[0] == 5.0
    assert select_frontier(clusters, 1, 0.0, 0.0)[0] == 1.0
    assert select_frontier(clusters, 2, 0.0, 0.0)[0] == 3.0
    assert select_frontier(clusters, 3, 0.0, 0.0) is None          # rank out of range
    assert select_frontier([], 0, 0.0, 0.0) is None


def test_min_distance_with_fallback():
    near, far = (0.2, 0.0, 0.0, 100), (2.0, 0.0, 0.0, 5)
    assert select_frontier([near, far], 0, 0.0, 0.0) == far          # closer than 0.4 m: skipped
    assert select_frontier([near], 0, 0.0, 0.0) == near              # unless nothing else is left


def grid_with_wall():
    g = np.zeros((40, 40), dtype=np.int8)                           # 0.1 m cells, 4 x 4 m
    g[:, 20] = 100                                                  # wall at x = 2.0-2.1 m
    g[:, 30:] = -1                                                  # unknown beyond x = 3 m
    return g


def test_snap_keeps_clearance_and_side():
    g = grid_with_wall()
    # goal right next to the wall, on its left side
    x, y = snap_to_clear_cell(g, (0.0, 0.0), 0.1, 1.95, 2.0, 0.15)
    assert x < 2.0                                                  # not moved through the wall
    ix, iy = int(x / 0.1), int(y / 0.1)
    r = math.ceil(0.15 / 0.1)
    assert (g[iy - r:iy + r + 1, ix - r:ix + r + 1] <= 50).all()
    # a centroid just inside unknown space (beside the frontier line) is pulled back to free space
    x, y = snap_to_clear_cell(g, (0.0, 0.0), 0.1, 3.05, 2.0, 0.15)
    assert 2.1 < x < 3.0 and g[int(y / 0.1), int(x / 0.1)] == 0
    # deep in unknown space: the search may not tunnel through it
    assert snap_to_clear_cell(g, (0.0, 0.0), 0.1, 3.8, 2.0, 0.15) is None


def test_pull_back():
    assert pull_back(0.5, 0.0, 0.0, 0.0) == (0.5, 0.0)              # close: unchanged
    x, y = pull_back(2.0, 0.0, 0.0, 0.0)
    assert abs(x - 1.7) < 1e-9 and y == 0.0


def test_reachable_mask_drops_frontiers_behind_walls():
    from trav_exploration.goal_selection import is_reachable, reachable_mask
    g = np.zeros((40, 80), dtype=np.int8)                      # two rooms, 0.05 m cells
    g[:, 38:42] = 100                                          # wall between them
    origin, res = (0.0, 0.0), 0.05
    m = reachable_mask(g, 0.5, 1.0, origin, res)
    assert is_reachable(m, 1.5, 1.0, origin, res)
    assert not is_reachable(m, 3.5, 1.0, origin, res)          # seen through a hole: left out
    g[18:22, 38:42] = 0                                        # a 0.2 m door: 4 cells
    assert is_reachable(reachable_mask(g, 0.5, 1.0, origin, res), 3.5, 1.0, origin, res)
    g[19:21, 38:42] = 100                                      # 1-cell gaps: inscribed for Nav2
    g[18, 38:42] = 0; g[21, 38:42] = 0
    assert not is_reachable(reachable_mask(g, 0.5, 1.0, origin, res), 3.5, 1.0, origin, res)
