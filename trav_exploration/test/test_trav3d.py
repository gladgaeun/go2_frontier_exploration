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


def test_climb_onto_the_landing_skips_a_failed_target():
    m, k, top = _stairs()
    g = _graph(m)
    edge = (0.6 + 0.3 * 6 - 0.05, 0.0, _z(m, top - 4))
    first, _ = g.climb(edge, radius=3.0)
    other, _ = g.climb(edge, radius=3.0, exclude=[first[-1]])
    assert other is None or np.linalg.norm(other[-1] - first[-1]) > 0.3


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


def test_reach_near_and_path_to_a_frontier_on_the_stairs():
    # the stair stage goes to the cheapest node near a semantic frontier (its voxels)
    m, k, top = _stairs()
    g = _graph(m)
    fs = np.array([[1.4, y, _z(m, k + 4 * 3)] for y in np.linspace(-0.3, 0.3, 7)])   # edge of the 3rd step
    node, J = g.reach_near((0.0, 0.0, _z(m, k)), fs)
    assert node >= 0 and np.isfinite(J)
    p = g.positions()[node]
    assert abs(p[2] - fs[0, 2]) <= 0.3 and abs(p[0] - 1.4) <= 0.5
    path = g.path_to((0.0, 0.0, _z(m, k)), node)
    assert path is not None and np.allclose(path[-1], p) and np.all(np.diff(path[:, 2]) <= 0.25 + 1e-6)
    assert g.reach_near((0.0, 0.0, _z(m, k)), fs + [0, 0, 5.0])[0] == -1                # nothing near: none
    # highest: the stand nearest to the frontier's height, not the cheap floor at the foot
    low = np.array([[0.7, 0.0, _z(m, k + 4)]])
    n_hi, _ = g.reach_near((0.0, 0.0, _z(m, k)), low, radius=0.4, dz=0.3, highest=True)
    n_lo, _ = g.reach_near((0.0, 0.0, _z(m, k)), low, radius=0.4, dz=0.3)
    assert g.positions()[n_hi][2] >= g.positions()[n_lo][2]
    assert g.positions()[n_hi][2] > _z(m, k) + 0.1


def test_climb_steps_off_the_top_step_when_nothing_higher_is_near_the_stairs():
    # the landing is known but not near the stairs in the graph (no stair voxel within one step):
    # from the top step the climb still steps off onto it, once
    m, k, top = _stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    land = xs >= 0.5 + 0.3 * 6                          # the landing beyond the last step
    m.stairs[land, :, :] = False
    m.trav_s[np.ix_(land, np.ones(m.ny, bool), [top])] = True
    m.reinflate_all()
    g = _graph(m)
    edge = (0.5 + 0.3 * 6 - 0.1, 0.0, _z(m, top - 4))   # on the last step
    assert g.climb(edge, radius=3.0)[0] is None or g.climb(edge, radius=3.0)[0][-1, 2] > edge[2]
    path, _ = g.climb(edge, radius=3.0, step_off=True)
    assert path is not None and path[-1, 2] >= edge[2] - m.res
    assert np.hypot(path[-1, 0] - edge[0], path[-1, 1] - edge[1]) >= 0.3
    off = path[-1]
    assert g.climb(off, radius=3.0, step_off=True) == (None, None) or \
        not g.near_stairs()[g.node(off, 0.6)]           # off the stairs: no further step-off


def test_descend_is_the_climb_mirrored():
    m, k, top = _stairs()
    g = _graph(m)
    landing = (2.6, 0.0, _z(m, top))
    path, _ = g.climb(landing, radius=3.5, down=True)
    assert path is not None and path[-1, 2] < landing[2] - 0.1          # down the stairs
    assert np.all(np.diff(path[:, 2]) >= -0.25 - 1e-6)                    # step by step
    p = path[-1]
    for _ in range(20):                                                   # stretch after stretch
        nxt, _ = g.climb(p, radius=3.5, down=True, step_off=True)
        if nxt is None:
            break
        p = nxt[-1]
    assert abs(p[2] - _z(m, k)) < 0.06                                    # on the floor below
    node, _ = g.reach_near(landing, np.array([[1.4, 0.0, _z(m, k + 12)]]), radius=0.6, lowest=True)
    assert node >= 0 and g.positions()[node][2] <= _z(m, k + 12) + 0.05


def test_stair_reach_bridges_treads_cut_short_by_a_mask():
    # every tread keeps only its front 0.15 m traversable (its back lies in O_inf below an
    # unlabelled nosing): 0.2 m gaps, which 2-cell neighbours do not bridge and 4-cell stair
    # neighbours do
    m, k, top = _stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    for n in range(6):
        x0 = 0.5 + 0.3 * n
        back = (xs >= x0 + 0.15) & (xs < x0 + 0.3)
        m.trav_s[back, :, k + 4 * (n + 1)] = False
    m.reinflate_all()
    snap = trav3d.SupportGraph.snapshot(m)
    start, goal = (0.0, 0.0, _z(m, k)), (2.6, 0.0, _z(m, top))
    assert trav3d.SupportGraph(snap, stair_reach=0).path(start, goal, goal_radius=0.3)[0] is None
    path, _ = trav3d.SupportGraph(snap, stair_reach=4).path(start, goal, goal_radius=0.3)
    assert path is not None and path[-1, 2] > _z(m, top) - 0.06


def test_snapshot_without_stair_margin_keeps_treads_next_to_o_inf():
    # an unlabelled nosing voxel on a tread: its O_inf and the one-cell margin around it take
    # the tread; without the stair margin only the O_inf of the layer is taken
    m, k, top = _stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    kk = k + 12
    i, j = int(np.argmin(np.abs(xs - 1.25))), int(np.argmin(np.abs(ys)))
    m.trav_s[i, j, kk] = False
    m.stairs[i, j, kk] = False
    m.reinflate_all()
    tight = trav3d.SupportGraph.snapshot(m)
    loose = trav3d.SupportGraph.snapshot(m, stair_margin=False)
    assert not (tight['trav'] & ~loose['trav']).any()                  # only adds
    extra = loose['trav'] & ~tight['trav']
    assert extra.any() and loose['stairs'][extra].all()                 # stair voxels only
    assert not (loose['trav'] & ~(m._source() & ~m.inflated)[tuple(
        slice(a, a + n) for a, n in zip(loose['lo'], loose['trav'].shape))]).any()   # never in O_inf


def test_step_off_the_bottom_step_onto_the_floor():
    # going down, the robot stopped on the first step above the floor: the next stretch leaves
    # the stairs onto the floor below, away from them
    m, k, top = _stairs()
    g = _graph(m)
    path, _ = g.step_off((0.65, 0.0, _z(m, k + 4)), down=True)
    assert path is not None
    end = path[-1]
    assert abs(end[2] - _z(m, k)) < 0.03                            # on the floor
    assert 0.4 <= np.hypot(end[0] - 0.65, end[1]) <= 2.0
    assert not g.near_stairs()[g.node(end, 0.05)]                   # off the stairs
    assert g.step_off((-1.0, 0.0, _z(m, k)), down=True)[0] is None    # on the floor: nothing lower


def test_flat_area_floor_landing_and_stairs():
    # a floor: metres of flat support connected to the robot; on a step: none (near the stairs);
    # the 0.7 m landing at the top of the stair scene: less than a floor (floor_min_area 3 m^2)
    m, k, top = _stairs()
    g = _graph(m)
    assert g.flat_area((-1.0, 0.0, _z(m, k))) >= 3.0
    assert g.flat_area((0.5 + 0.3 * 2 + 0.15, 0.0, _z(m, k + 12))) == 0.0
    landing = g.flat_area((2.75, 0.0, _z(m, top)))
    assert 0.0 <= landing < 3.0


def _unlabel_steps(m, steps):
    """Steps the segmentation missed: occupied, but not in M_s nor stair-labelled."""
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    for n in steps:
        sel = (xs >= 0.5 + 0.3 * n) & (xs < 0.8 + 0.3 * n)
        m.trav_s[sel] = False
        m.stairs[sel] = False
        m.stair_seen[sel] = False
    m.stair_xy = m.centers(np.flatnonzero(m.stairs))[:, :2]
    m.reinflate_all()


def _seen_free(m):
    """Everything not occupied was seen free (the camera looked through the stairwell)."""
    m.logodds[m.logodds == 0.0] = m.l_min


def test_geometric_treads_bridge_unlabelled_steps():
    m, k, top = _stairs()
    _seen_free(m)
    m.stair_seen |= m.stairs
    _unlabel_steps(m, (2, 3))
    start = (0.75, 0.0, _z(m, k + 4))                       # on step 0
    before, _ = _graph(m).climb(start, radius=3.0)
    hi_before = before[-1, 2] if before is not None else start[2]
    assert hi_before < _z(m, k + 4 * 3) - 1e-6             # stuck below the unlabelled steps
    add, steps = m.geometric_treads(0.75, 0.0, _z(m, k + 4), radius=2.5, up=2.0)
    assert steps.size and np.isin(steps, add).all()
    ii, jj, kk = np.unravel_index(add, m.logodds.shape)
    assert add.size and set(np.unique(kk)) & {k + 4 * 3, k + 4 * 4}       # steps 2 and 3
    assert (np.abs(m.origin_xy + (jj + 0.5) * m.res) < 0.6).all()      # not the side walls
    m.integrate_semantic(m.centers(add))
    after, _ = _graph(m).climb(start, radius=3.0)
    assert after is not None and after[-1, 2] > hi_before + 0.3


def test_geometric_treads_leave_furniture_away_from_stairs():
    m, k, top = _stairs()
    _seen_free(m)
    m.stair_seen |= m.stairs
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    box = np.ix_((xs > -2.0) & (xs < -1.4), (ys > -0.3) & (ys < 0.3), range(k, k + 4))
    m.logodds[box] = m.l_max                                 # an ottoman, 0.15-0.2 m high
    m.reinflate_all()
    add, _ = m.geometric_treads(-1.0, 0.0, _z(m, k), radius=2.0)
    ii, _, _ = np.unravel_index(add, m.logodds.shape)
    assert not (m.origin_xy + (ii + 0.5) * m.res < -1.3).any()


def test_geometric_treads_leave_the_seen_top_of_a_wall():
    # the side wall of the stairwell seen only up to some height along the flight (unknown above,
    # free in front of it): its observed upper edge is no tread, whatever its height steps
    m, k, top = _stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    wall = np.abs(ys) >= 0.6
    m.logodds[np.ix_(xs > 0.4, wall, range(k, m.nz))] = 0.0       # wall unseen ...
    for n in range(6):                                            # ... except up to 0.4 m over each step
        sel = (xs >= 0.5 + 0.3 * n) & (xs < 0.8 + 0.3 * n)
        kk = k + 4 * (n + 1)
        m.logodds[np.ix_(sel, np.abs(ys - 0.625) < 0.01, range(k, kk + 8))] = m.l_max
    inside = np.abs(ys) < 0.6                                     # the stairwell itself seen free
    air = np.zeros(m.logodds.shape, dtype=bool)
    air[:, inside, :] = True
    m.logodds[air & (m.logodds == 0.0)] = m.l_min
    m.stair_seen |= m.stairs
    m.reinflate_all()
    add, steps = m.geometric_treads(0.75, 0.0, _z(m, k + 4), radius=2.5, up=2.0)
    _, jj, _ = np.unravel_index(add, m.logodds.shape)
    assert not (np.abs(ys[jj]) >= 0.6).any()                      # nothing on the wall


def test_cluster_costs_min_over_voxels_and_none_behind_a_wall():
    m, k, top = _stairs()
    g = _graph(m)
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    # a cluster on the floor in front of the stairs, and one on the floor behind the side wall of
    # the stairwell (|y| >= 0.6 is wall from x > 0.4 up), next to a reachable cell
    near = [np.ravel_multi_index((i, int(np.argmin(np.abs(ys))), k), m.logodds.shape)
            for i in np.flatnonzero((xs > -1.0) & (xs < -0.5))]
    j_in = int(np.argmin(np.abs(ys - 0.55)))
    behind = [np.ravel_multi_index((i, jj, k), m.logodds.shape)
              for i in np.flatnonzero((xs > 1.0) & (xs < 1.2)) for jj in (j_in + 3, j_in + 4)]
    (J1, t1), (J2, t2) = g.cluster_costs((-2.0, 0.0, _z(m, k)), [near, behind], m.logodds.shape)
    assert np.isfinite(J1) and t1 is not None and t1[0] < -0.45   # its cheapest voxel: the near end
    assert not np.isfinite(J2) and t2 is None


def test_geometric_steps_leave_the_floor_beyond_unlabelled():
    m, k, top = _stairs()
    _seen_free(m)
    m.stair_seen |= m.stairs
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    far = xs >= 0.5 + 0.3 * 2                               # steps 2.. and the landing unlabelled
    m.trav_s[far] = False
    m.stairs[far] = False
    m.stair_seen[far] = False
    m.stair_xy = m.centers(np.flatnonzero(m.stairs))[:, :2]
    m.reinflate_all()
    add, steps = m.geometric_treads(0.95, 0.0, _z(m, k + 4 * 2), radius=3.0, up=1.5,
                                    hint=(1.5, 0.0, 2.0))
    ia, _, ka = np.unravel_index(add, m.logodds.shape)
    ist, _, kst = np.unravel_index(steps, m.logodds.shape)
    landing_end = 0.5 + 0.3 * 6
    assert ((ka == top) & (xs[ia] > landing_end + 0.6)).any()          # the landing is support
    assert not ((kst == top) & (xs[ist] > landing_end + 0.5)).any()     # but no step past its edge
    assert ((kst < top) & (kst > k + 4 * 2)).any()                     # the steps up to it are


def _object(x, y, z0=0.0, r=0.15, h=0.5):
    """Points of a small object (a chair) standing at (x, y) on the floor at z0."""
    a = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    zz = np.linspace(z0, z0 + h, 5)
    return np.array([(x + r * np.cos(t), y + r * np.sin(t), z) for t in a for z in zz])


def test_target_goal_is_a_reachable_node_near_the_object():
    m = _room()
    g = _graph(m)
    pts = _object(-1.0, 2.2)
    r = g.target_goal((0.0, 0.0, 0.0), pts, r_min=0.35, r_goal=0.75, k_min=3)
    assert r['node'] >= 0 and np.isfinite(r['J'])
    p = g.positions()[r['node']]
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    assert 0.35 <= d.min() <= 0.75 and (d <= 0.75).sum() >= 3
    assert abs(p[2] - _z(m, m.layer(0.0))) <= m.res      # on the floor, at the floor's own height
    # the cheapest of the candidates
    assert r['n_reach'] >= 1 and r['n_clear'] >= r['n_reach']


def test_target_goal_counts_where_the_candidates_run_out():
    m = _room()
    g = _graph(m)
    r = g.target_goal((0.0, 0.0, 0.0), _object(4.0, 4.0), r_goal=0.75)    # outside the room
    assert r['node'] < 0 and r['n_near'] == 0
    # an object on a floor 3 m higher: none of this floor's nodes is a goal for it
    r = g.target_goal((0.0, 0.0, 0.0), _object(-1.0, 2.2, z0=3.0), r_goal=0.75)
    assert r['node'] < 0 and r['n_near'] == 0


def test_target_goal_keeps_the_goal_clearance():
    m = _room()
    g = _graph(m)
    pts = _object(-1.0, 2.2)
    r = g.target_goal((0.0, 0.0, 0.0), pts, r_goal=0.75, min_d_obs=m.robot_radius + 0.2)
    if r['node'] >= 0:
        assert g.d_obs[r['node']] >= m.robot_radius + 0.2
    r_all = g.target_goal((0.0, 0.0, 0.0), pts, r_goal=0.75)
    assert r['n_clear'] <= r_all['n_clear']


def test_clear_node_moves_a_goal_off_o_inf():
    m = _room()
    g = _graph(m)
    need = m.robot_radius + 0.2
    tight = np.flatnonzero(g.d_obs < need)
    pos = g.positions()
    moved = 0
    for k in tight[::max(1, len(tight) // 20)]:
        c = g.clear_node(int(k), need, radius=0.5)
        if c >= 0:
            assert g.d_obs[c] >= need
            assert np.hypot(*(pos[c, :2] - pos[k, :2])) <= 0.5 + 1e-6
            moved += 1
    assert moved > 0
    clear = int(np.flatnonzero(g.d_obs >= need)[0])
    assert g.clear_node(clear, need) == clear                 # a clear node stays
