# Unit tests for collision_guard (pure numpy, no ROS).
#   python3 -m pytest trav_exploration/test/test_collision_guard.py

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.collision_guard import (AbortCounter, Gate, Trail, danger_points, follow,  # noqa: E402
                                             front_row, next_target)


def test_danger_zone():
    pts = [(0.4, 0.0),     # straight ahead within radius + distance: danger
           (0.4, 0.24),    # grazing the body side: danger
           (0.4, 0.26),    # passes beside the robot
           (0.6, 0.0),     # farther than 0.25 + 0.3
           (-0.2, 0.0)]    # behind
    d = danger_points(pts, radius=0.25, distance=0.3)
    assert d.tolist() == [[0.4, 0.0], [0.4, 0.24]]
    assert danger_points(np.zeros((0, 2)), 0.25, 0.3).shape == (0, 2)


def test_gate_stops_forward_only_and_holds():
    g = Gate(hold=1.0)
    assert g.update(0.0, True)                  # rising edge: a new stop
    assert not g.update(0.1, True)              # still the same stop
    assert g.filter(0.2, 0.4, 0.5) == (0.0, 0.5)     # no forward, turning allowed
    assert g.filter(0.2, -0.1, 0.0) == (-0.1, 0.0)   # reversing allowed
    assert not g.update(0.5, False)
    assert g.filter(1.05, 0.4, 0.0) == (0.0, 0.0)    # within hold after the last danger
    assert g.filter(1.2, 0.4, 0.0) == (0.4, 0.0)     # released
    assert g.update(2.0, True)                  # a new stop later


def test_front_row():
    row = front_row(1.0, 2.0, math.pi / 2, 0.25, 0.05, 0.32)
    assert np.allclose(row[:, 1], 2.0 + 0.32)   # in front of the robot, facing +y
    assert np.isclose(row[:, 0].min(), 0.75) and np.isclose(row[:, 0].max(), 1.25)


def test_abort_counter():
    c = AbortCounter(n=3, window=20.0, radius=0.3)
    assert not c.abort(0.0, 0.0, 0.0)
    assert not c.abort(5.0, 0.1, 0.0)
    assert c.abort(10.0, 0.0, 0.1)              # third abort at the same spot
    assert not c.abort(11.0, 0.0, 0.0)          # counting restarts
    assert not c.abort(12.0, 2.0, 0.0)          # elsewhere: does not add up
    assert not c.abort(40.0, 0.0, 0.0)          # too late for the window


def test_block_does_not_trap_the_robot():
    from trav_exploration.voxel_mapper import VoxelMapper
    m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
    k = m.layer(0.0)
    m.trav_g[:, :, k] = True                                   # open floor
    ijk = lambda x, y: (int((x - m.origin_xy) / m.res), int((y - m.origin_xy) / m.res), k)  # noqa: E731
    robot = (0.025, 0.025)
    m.mark_blocked(front_row(*robot, 0.0, 0.25, 0.05, 0.32), 0.0)
    assert m.traversable()[ijk(*robot)]                        # the robot can still turn away
    assert not m.traversable()[ijk(0.325, 0.025)]              # the way ahead is closed
    # the old placement (just past the body, 0.275 m) put the robot inside O_inf
    m2 = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
    m2.trav_g[:, :, k] = True
    m2.mark_blocked(front_row(*robot, 0.0, 0.25, 0.05, 0.275), 0.0)
    assert not m2.traversable()[ijk(robot[0] + 0.05, robot[1])]


def test_block_never_closes_the_way_the_robot_came():
    from trav_exploration.voxel_mapper import VoxelMapper
    m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
    k = m.layer(0.0)
    m.trav_g[:, :, k] = True
    ijk = lambda x, y: (int((x - m.origin_xy) / m.res), int((y - m.origin_xy) / m.res), k)  # noqa: E731
    for x in np.arange(-1.0, 0.01, 0.2):                      # the robot drove from x = -1 to x = 0
        m.visit(x, 0.0)
    behind = front_row(0.0, 0.0, math.pi, 0.25, 0.05, 0.37)   # a failure blamed on the way back
    ahead = front_row(0.0, 0.0, 0.0, 0.25, 0.05, 0.37)
    assert m.mark_blocked(behind, 0.0) == 0
    assert m.mark_blocked(ahead, 0.0) > 0
    for x in np.arange(-1.0, 0.01, 0.05):                     # the way back stays traversable
        assert m.traversable()[ijk(x, 0.0)]
    assert not m.traversable()[ijk(0.35, 0.0)]
    # a block beside the path (0.3 m off it) would leave a 1-2 cell gap that Nav2's own inflation
    # closes (the robot was trapped this way in a corridor): skipped as well
    assert m.mark_blocked(np.array([[-0.5, 0.3]]), 0.0) == 0
    assert m.mark_blocked(np.array([[-0.5, 0.4]]), 0.0) == 1


def test_robot_is_never_inside_obstacles_on_the_nav2_grid():
    from trav_exploration.voxel_mapper import VoxelMapper
    m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
    k = m.layer(0.0)
    m.trav_g[:, :, k] = True
    for x in np.arange(-1.0, 0.01, 0.2):
        m.visit(x, 0.0)
    (i, j, kk), = m._ijk([0.0, 0.2, m.origin_z + (k + m.dz_lo + 0.5) * m.res])[0]
    m.logodds[i, j, kk] = m.l_max                              # an obstacle seen 0.2 m beside the robot
    m.reinflate_all()
    assert not m.traversable()[int((0.0 - m.origin_xy) / m.res), int((0.0 - m.origin_xy) / m.res), k]
    data, w, h, ox, oy = m.trav_grid(0.0)
    g = data.reshape(h, w)
    cell = lambda x, y: g[int((y - oy) / m.res), int((x - ox) / m.res)]  # noqa: E731
    for x in np.arange(-1.0, 0.01, 0.05):                     # its path stays free for Nav2
        assert cell(x, 0.0) == m.FREE and cell(x, 0.05) == m.FREE
    assert cell(0.0, 0.1) == m.OCC                             # the rest of O_inf does not


def test_one_stuck_voxel_closes_a_narrow_gap_but_not_a_hallway():
    from trav_exploration.voxel_mapper import VoxelMapper
    for width, closed in ((0.2, True), (0.8, False)):          # free width on the map (after O_inf)
        m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
        k = m.layer(0.0)
        y = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
        m.trav_g[:, np.abs(y) < width / 2, k] = True           # a corridor along x
        m.mark_blocked(front_row(0.0, 0.0, 0.0, 0.0, 0.05, 0.37), 0.0)
        i = int((0.37 - m.origin_xy) / m.res)
        assert (not m.traversable()[i, :, k].any()) == closed


def test_trail_back_and_follow():
    tr = Trail(spacing=0.1)
    for x in np.arange(0.0, 2.001, 0.02):                      # drove along x, then turned up y
        tr.add(x, 0.0)
    for y in np.arange(0.0, 1.001, 0.02):
        tr.add(2.0, y)
    wps = tr.back(2.0, 1.0, 1.5)
    assert 1.5 <= math.hypot(wps[-1][0] - 2.0, wps[-1][1] - 1.0) < 1.75   # straight-line 1.5 m (+ a step)
    assert np.isclose(wps[-1][1], 0.0)                                                 # on the x leg
    assert all(math.hypot(a[0] - b[0], a[1] - b[1]) < 0.25 for a, b in zip(wps, wps[1:]))  # crumbs 0.1-0.2 m apart
    assert follow(0.0, 0.0, 0.0, (1.0, 0.0))[0] > 0.0          # ahead: drive
    vx, wz = follow(0.0, 0.0, 0.0, (-1.0, 0.0))                  # behind: turn in place first
    assert vx == 0.0 and wz != 0.0


def test_escape_leaves_a_jitter_cluster():
    rng = np.random.default_rng(1)
    tr = Trail(spacing=0.1)
    for x in np.arange(-1.5, 0.001, 0.02):                     # came in along x
        tr.add(x, 0.0)
    for _ in range(300):                                       # then jittered around the origin
        tr.add(*rng.uniform(-0.15, 0.15, 2))
    wps = tr.back(0.0, 0.0, 1.0)
    assert math.hypot(*wps[-1]) >= 1.0 and wps[-1][0] < -0.9   # back out the way it came
    # unicycle simulation with the escape controller
    x, y, yaw, i = 0.0, 0.0, 0.0, 0
    for _ in range(400):
        i = next_target(wps, i, x, y)
        if math.hypot(wps[-1][0] - x, wps[-1][1] - y) < 0.1:
            break
        vx, wz = follow(x, y, yaw, wps[i])
        yaw += wz * 0.1
        x += vx * 0.1 * math.cos(yaw)
        y += vx * 0.1 * math.sin(yaw)
    assert math.hypot(wps[-1][0] - x, wps[-1][1] - y) < 0.1    # reached within 40 s


def test_guard_stop_closes_a_gap_too_narrow_for_the_body_but_not_a_doorway():
    from trav_exploration.voxel_mapper import VoxelMapper
    for width, closed in ((0.6, False), (0.45, True)):          # wall at x = 0.5 with a gap at y = 0
        m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
        k = m.layer(0.0)
        m.trav_g[:, :, k] = True
        xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
        ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
        wall_i = np.flatnonzero(np.abs(xs - 0.525) < 0.03)
        wall_j = np.flatnonzero(np.abs(ys) > width / 2)
        m.logodds[np.ix_(wall_i, wall_j, [k + m.dz_lo])] = m.l_max
        m.reinflate_all()
        i = int((0.525 - m.origin_xy) / m.res)
        free_before = m.traversable()[i, :, k].any()
        assert free_before                                      # O_inf (0.22) leaves both open
        m.visit(-0.5, 0.0)
        m.visit(0.0, 0.0)
        m.mark_tight(0.0, 0.0, 0.0, 0.0, radius=0.25)
        data, w, h, ox, oy = m.trav_grid(0.0)
        g = data.reshape(h, w)
        col = g[:, int((0.525 - ox) / m.res)]
        assert (not (col == m.FREE).any()) == closed
        assert g[int((0.0 - oy) / m.res), int((0.0 - ox) / m.res)] == m.FREE   # the robot's own cell


def test_tight_after_contact_keeps_the_robot_free_to_turn():
    # stuck against a wall 0.25 m ahead (touching it): the cells next to the robot are within the
    # body radius of the wall too, but closing them leaves the robot's cell INSCRIBED for Nav2
    from scipy import ndimage
    from trav_exploration.voxel_mapper import VoxelMapper
    m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22)
    k = m.layer(0.0)
    m.trav_g[:, :, k] = True
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    m.logodds[np.ix_(np.flatnonzero(np.abs(xs - 0.275) < 0.03), np.arange(m.ny), [k + m.dz_lo])] = m.l_max
    m.reinflate_all()
    m.visit(-0.5, 0.0)
    m.visit(0.0, 0.0)
    assert m.mark_tight(0.0, 0.0, 0.0, 0.0, radius=0.25) > 0
    data, w, h, ox, oy = m.trav_grid(0.0)
    g = data.reshape(h, w)
    inscribed = ndimage.binary_dilation(g == m.OCC, structure=ndimage.generate_binary_structure(2, 1))
    r, c = int((0.0 - oy) / m.res), int((0.0 - ox) / m.res)
    assert not inscribed[r, c]                                   # the robot can spin and back up
    assert not inscribed[r, c - 1]                               # and back away from the wall
    cx = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    near = np.hypot(cx[:, None], cx[None, :]) < 0.15
    assert not (m.tight & near).any()                            # no tight cell next to the robot


def test_trail_back_does_not_jump_to_an_old_pass():
    tr = Trail(spacing=0.1)
    for x in np.arange(0.0, 3.001, 0.02):                      # long ago: along y = 0 from x = 0 to 3
        tr.add(x, 0.0)
    for y in np.arange(0.0, 20.001, 0.02):                     # then far away and back (>300 crumbs)
        tr.add(3.0, y)
    for y in np.arange(20.0, -0.001, -0.02):
        tr.add(3.05, y)
    for x in np.arange(3.0, 1.499, -0.02):                     # now along y = 0 the other way, to x = 1.5
        tr.add(x, 0.05)
    wps = tr.back(1.5, 0.05, 1.0)
    assert wps[-1][0] > 2.4                                    # back where it just came from (x grows)
    assert sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:])) < 1.5
