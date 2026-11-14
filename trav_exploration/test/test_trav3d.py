# Unit tests for the 3-D path search on support voxels (trav3d.py; pure numpy, no ROS).
#   python3 -m pytest trav_exploration/test/test_trav3d.py

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration import trav3d  # noqa: E402
from trav_exploration.voxel_mapper import VoxelMapper  # noqa: E402


def _stairs(n_steps=6, gap_step=None):
    """Floor at layer k, then n_steps steps of 0.2 m (4 layers) x 0.3 m along x, a landing on top;
    support voxels go into M_s (mode s), the stair ones into the stair voxels."""
    m = VoxelMapper(resolution=0.05, size_xy=6.0, z_above=3.0, robot_radius=0.22, mode='s')
    k = m.layer(0.0)
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    width = np.abs(ys) < 0.6
    m.logodds[:, :, k] = m.l_max
    m.trav_s[:, :, k] = True
    top = k
    for n in range(n_steps + 1):                       # the last one is the landing
        # only surfaces are seen (the inside of a step stays unknown): the riser at the front
        # column and the tread on top, both labelled stairs (in M_s)
        x0 = 0.5 + 0.3 * n
        sel = (xs >= x0) & (xs < (x0 + 0.3 if n < n_steps else 3.0))
        front = np.flatnonzero(sel)[:1]
        kk = k + 4 * (n + 1)
        if gap_step == n:
            continue                                   # step not seen (FOV gap)
        stair = n < n_steps
        m.logodds[np.ix_(front, width, range(kk - 4, kk + 1))] = m.l_max          # riser
        m.trav_s[np.ix_(front, width, range(kk - 4, kk + 1))] = True
        m.stairs[np.ix_(front, width, range(kk - 4, kk + 1))] = stair
        m.logodds[np.ix_(sel, width, [kk])] = m.l_max                             # tread
        m.trav_s[np.ix_(sel, width, [kk])] = True
        m.stairs[np.ix_(sel, width, [kk])] = stair
        top = kk
    m.stair_xy = m.centers(np.flatnonzero(m.stairs))[:, :2]
    wall = np.abs(ys) >= 0.6
    m.logodds[np.ix_(xs > 0.4, wall, range(k, m.nz))] = m.l_max          # side walls of the stairwell
    m.reinflate_all()
    return m, k, top


def _graph(m, **kw):
    return trav3d.SupportGraph(trav3d.SupportGraph.snapshot(m, **kw))


def _z(m, k):
    return m.origin_z + (k + 0.5) * m.res


def test_graph_climbs_the_stairs():
    # mode s: the floors are connected over the stairs ("connectivity across stairs and floors")
    m, k, top = _stairs()
    start, goal = (0.0, 0.0, _z(m, k)), (2.6, 0.0, _z(m, top))
    path, cost = _graph(m).path(start, goal, goal_radius=0.3)
    assert path is not None
    assert abs(path[-1, 2] - goal[2]) < 1e-6 and np.all(np.diff(path[:, 2]) <= 0.25 + 1e-6)   # step by step
    assert np.all(np.abs(path[:, 1]) < 0.6 - 0.22)     # off the walls (O_inf)
    assert cost >= np.hypot(2.6, goal[2] - start[2])


def test_graph_path_on_the_floor_stays_on_the_floor():
    m, k, _ = _stairs()
    path, _ = _graph(m).path((0.0, 0.0, _z(m, k)), (0.3, 0.3, _z(m, k)))
    assert path is not None and np.all(np.abs(path[:, 2] - _z(m, k)) < 0.06)


def test_graph_bridges_a_small_gap_but_not_a_missing_step():
    m, k, top = _stairs(gap_step=2)                    # one tread not observed: 0.3 m gap, 0.4 m rise
    assert _graph(m).path((0.0, 0.0, _z(m, k)), (2.6, 0.0, _z(m, top)), goal_radius=0.3)[0] is None


def test_graph_in_mode_g_does_not_cross_stairs():
    m, k, top = _stairs()
    m.set_mode('g')                                    # T = M_g: stairs are obstacles (eq. 1)
    m.trav_g[:, :, k] = True
    m.reinflate_all()
    assert _graph(m).path((0.0, 0.0, _z(m, k)), (2.6, 0.0, _z(m, top)), goal_radius=0.3)[0] is None


def test_graph_node_stays_at_the_given_height():
    # at the foot of the stairs the first tread (0.2 m up) is nearer than the floor in 3-D; a
    # point on the floor must snap to the floor (a start on the tread found no floor frontier)
    m, k, _ = _stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    m.trav_s[np.abs(xs - 0.65) < 0.5, :, k - 1:k + 4] = False   # no floor / riser foot mapped there
    m.reinflate_all()
    g = _graph(m)
    p = (0.62, 0.0, _z(m, k))
    n = g.node(p, 0.6)
    assert n >= 0 and abs(g.positions()[n, 2] - _z(m, k)) < 0.06
    assert g.positions()[g.node(p, 0.6, dz=1.0), 2] > _z(m, k) + 0.1      # without the limit: the tread


def test_climb_is_the_highest_reachable_stair_voxel():
    m, k, top = _stairs()
    g = _graph(m)
    path, _ = g.climb((0.0, 0.0, _z(m, k)), radius=3.0)
    assert path is not None
    assert path[-1, 2] >= _z(m, k + 4 * 6) - 1e-6      # top step or the landing next to it
    assert g.climb(path[-1], radius=3.0) == (None, None)       # already at the top


def test_climb_from_afar_goes_over_the_floor_to_the_stairs():
    # one path from the robot over the floor and up (centre = the foot of the stairs)
    m, k, top = _stairs()
    path, _ = _graph(m).climb((-2.0, 1.5, _z(m, k)), center=(0.5, 0.0, _z(m, k)), radius=3.0)
    assert path is not None and path[-1, 2] >= _z(m, k + 4 * 6) - 1e-6
    assert np.hypot(path[0, 0] + 2.0, path[0, 1] - 1.5) < 0.1


def test_climb_steps_off_the_top_onto_the_landing():
    m, k, top = _stairs()
    g = _graph(m)
    edge = (0.6 + 0.3 * 6 - 0.05, 0.0, _z(m, top - 4))   # on the last step, at its end
    path, _ = g.climb(edge, radius=3.0)
    assert path is not None and abs(path[-1, 2] - _z(m, top)) < 1e-6
    assert np.hypot(path[-1, 0] - edge[0], path[-1, 1] - edge[1]) >= 0.3     # onto the landing
    assert g.climb(path[-1], radius=3.0) == (None, None)


def test_climb_skips_a_failed_target():
    m, k, top = _stairs()
    g = _graph(m)
    start = (0.0, 0.0, _z(m, k))
    first, _ = g.climb(start, radius=3.0)
    other, _ = g.climb(start, radius=3.0, exclude=[first[-1]])
    assert other is not None and np.linalg.norm(other[-1] - first[-1]) > 0.3


def test_stair_landmark_has_entrance_at_the_foot_and_top_above():
    m, k, top = _stairs()
    m.stair_seen |= m.stairs
    lms = m.stair_landmarks(min_voxels=50)
    assert len(lms) == 1
    ex, ey, ez, tx, ty, tz, n = lms[0]
    assert ex < 0.8 and abs(ey) < 0.1 and abs(ez - (m.origin_z + (k + 0.5) * m.res)) < 0.06   # foot of the stairs
    assert tx > 2.0 and tz > ez + 1.0                                                          # top, one floor up
    assert m.stair_landmarks(min_voxels=10 ** 6) == []                                         # too small: not reliable


def _room():
    """The synthetic room of test_voxel_mapper (walls, a table, a low box), seen all around."""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_voxel_mapper import build
    return build()


def test_floor_graph_plans_around_obstacles_on_m_trav():
    m = _room()
    g = _graph(m)
    path, cost = g.path((0.0, 0.0, 0.0), (-1.0, 2.2, 0.0))
    assert path is not None and np.isfinite(cost)
    assert np.linalg.norm(path[-1, :2] - [-1.0, 2.2]) < 0.3
    trav = m.traversable()
    for p in path:                                     # every step on M_trav (not in O_inf)
        ijk = np.floor((p - [m.origin_xy, m.origin_xy, m.origin_z]) / m.res).astype(int)
        assert trav[tuple(ijk)]
    # J_geo from the robot equals the path cost; outside the room there is no path
    C = g.costs([(0.0, 0.0, 0.0)], [(-1.0, 2.2, 0.0), (4.0, 4.0, 0.0)])
    assert abs(C[0, 0] - cost) < 0.5 and np.isinf(C[0, 1])


def test_floor_graph_keeps_the_robots_own_cells():
    m = _room()
    k = m.layer(0.0)
    ix = int(np.floor((0.0 - m.origin_xy) / m.res))
    m.trav_g[ix - 3:ix + 4, ix - 3:ix + 4, k] = False  # the cells around the robot dropped out
    visited = np.zeros((m.nx, m.ny), dtype=bool)
    visited[ix - 3:ix + 4, ix - 3:ix + 4] = True
    assert _graph(m).node((0.0, 0.0, 0.0), 0.2) < 0
    g = _graph(m, visited={k: visited})
    assert g.node((0.0, 0.0, 0.0), 0.2) >= 0
    assert g.path((0.0, 0.0, 0.0), (-1.0, -1.5, 0.0))[0] is not None


def test_floor_graph_path_keeps_a_cell_off_o_inf():
    m = _room()
    g = _graph(m)
    path, _ = g.path((0.0, 0.0, 0.0), (-1.0, 2.2, 0.0))
    k = m.layer(0.0)
    for p in path:
        i, j = np.floor((p[:2] - m.origin_xy) / m.res).astype(int)
        for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            assert not m.inflated[i + di, j + dj, k - 1:k + 2].any()


def test_floor_graph_path_keeps_off_o_inf_where_there_is_room():
    # eq. 16 with d_obs from O_inf: in open space the path stays clear of O_inf by more than the
    # one-cell margin (0.1 m cells here), not along its edge
    m = _room()
    snap = trav3d.SupportGraph.snapshot(m)
    g = trav3d.SupportGraph(snap, lambda_esdf=0.3, d_safe=0.6)
    path, _ = g.path((0.0, -1.0, 0.0), (0.0, 2.3, 0.0))      # past the table's O_inf on its left
    d = snap['d_obs'][:, :, m.layer(0.0)]
    ij = np.floor((path[:, :2] - snap['origin'][:2]) / m.res).astype(int) - snap['lo'][:2]
    assert np.median(d[ij[:, 0], ij[:, 1]]) > m.robot_radius + 2 * m.res
