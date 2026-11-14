# Unit tests for the semantic mode (semantic.py, VoxelMapper M_s; pure numpy, no ROS).
#   python3 -m pytest trav_exploration/test/test_semantic.py

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.semantic import FLOOR, OTHER, RUG, STAIRS, ModeSwitch, split_classes  # noqa: E402
from trav_exploration.voxel_mapper import VoxelMapper  # noqa: E402


def test_split_classes():
    trav, stairs = split_classes([OTHER, FLOOR, RUG, STAIRS])
    assert trav.tolist() == [False, True, True, True]
    assert stairs.tolist() == [False, False, False, True]


def test_mode_switch_hysteresis():
    sw = ModeSwitch(enter_time=0.5, exit_time=2.0, min_stair_points=10, enter_range=4.0, exit_distance=5.0)  # explicit values
    t = iter(np.arange(0.0, 100.0, 0.05))                # camera frames at 20 Hz
    assert not sw.update(next(t), 20, np.inf)            # 0.00 s: first stair frame
    assert not sw.update(next(t), 20, np.inf)
    assert not sw.update(next(t), 0, np.inf)             # a frame without stairs restarts it
    changed = [sw.update(next(t), 20, np.inf) for _ in range(11)]
    assert changed == [False] * 10 + [True] and sw.mode == 's'   # 0.5 s of stair frames
    far = [sw.update(next(t), 0, 6.0) for _ in range(30)]
    assert not any(far)                                  # 1.5 s far: not yet
    assert not sw.update(next(t), 0, 4.5)                # back near the stairs: restarts
    far = [sw.update(next(t), 0, 6.0) for _ in range(41)]
    assert far[-1] and not any(far[:-1]) and sw.mode == 'g'      # 2.0 s far


def test_few_stair_pixels_do_not_switch():
    sw = ModeSwitch(enter_time=0.5, min_stair_points=50)
    for k in range(200):
        assert not sw.update(0.05 * k, 30, np.inf)       # a glimpse of a few stair pixels
    assert sw.mode == 'g'
    assert sw.request('s') and sw.mode == 's'            # planner request (stair landmark)


def _step_scene():
    """Floor at layer k; a 0.2 m step (4 voxel layers) at x in [0.5, 0.8)."""
    m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22, h_min=0.15, h_max=0.6)
    k = m.layer(0.0)
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    floor = np.ones(m.nx, bool)
    step = (xs >= 0.5) & (xs < 0.8)
    m.logodds[:, :, k] = m.l_max
    m.trav_g[np.ix_(floor & ~step, np.arange(m.ny), [k])] = True       # 2-D ray casting: floor only
    m.logodds[np.ix_(step, np.arange(m.ny), range(k + 1, k + 5))] = m.l_max
    m.reinflate_all()
    grid = np.stack(np.meshgrid(np.flatnonzero(step), np.arange(m.ny), range(k + 1, k + 5), indexing='ij'), -1)
    step_pts = m.centers(np.ravel_multi_index(grid.reshape(-1, 3).T, m.logodds.shape))
    gi = np.stack(np.meshgrid(np.flatnonzero(~step), np.arange(m.ny), [k], indexing='ij'), -1)
    floor_pts = m.centers(np.ravel_multi_index(gi.reshape(-1, 3).T, m.logodds.shape))
    return m, k, xs, step_pts, floor_pts


def test_stairs_become_traversable_in_semantic_mode_only():
    m, k, xs, step_pts, floor_pts = _step_scene()
    i_before = int(np.flatnonzero(xs > 0.4)[0])          # floor 0.1 m in front of the step
    i_top = int(np.flatnonzero(xs > 0.6)[0])             # middle of the step
    j = m.ny // 2
    assert not m.traversable()[i_before, j, k]           # g: the riser inflates the floor before it
    m.integrate_semantic(np.concatenate([floor_pts, step_pts]), step_pts)
    assert not m.traversable()[i_before, j, k]           # collected, but g still uses M_g
    assert m.stair_distance(0.0, 0.0) < 0.55 and m.stair_distance(-1.5, 0.0) > 1.9
    m.set_mode('s')
    assert m.traversable()[i_before, j, k]               # s: step voxels are support, not obstacles
    assert m.traversable()[i_top, j, k + 4]              # top of the step is traversable
    m.set_mode('g')
    assert not m.traversable()[i_before, j, k]


def test_semantic_insert_in_s_mode_updates_inflation_incrementally():
    m, k, xs, step_pts, floor_pts = _step_scene()
    m.integrate_semantic(floor_pts)
    m.set_mode('s')
    i_before = int(np.flatnonzero(xs > 0.4)[0])
    j = m.ny // 2
    assert not m.traversable()[i_before, j, k]           # step not labelled yet: obstacle
    m.integrate_semantic(step_pts, step_pts)             # now seen as stairs
    assert m.traversable()[i_before, j, k]
    full = m.inflated.copy()
    m.reinflate_all()
    assert (full == m.inflated).all()                    # incremental = global inflation


def test_trusted_stairs_keep_the_first_steps_and_drop_the_railing():
    m = VoxelMapper(resolution=0.05, size_xy=4.0, robot_radius=0.22, h_min=0.15, h_max=0.6)
    z_r = 0.0
    xs = np.arange(0.5, 1.5, 0.02)
    steps = []
    for n in range(8):                                  # 8 steps of 0.17 m x 0.25 m along x
        x0 = 0.5 + 0.25 * n
        for y in np.arange(-0.4, 0.41, 0.05):
            for x in np.arange(x0, x0 + 0.25, 0.05):
                steps.append((x, y, 0.17 * (n + 1)))    # tread
            for z in np.arange(0.17 * n, 0.17 * (n + 1), 0.04):
                steps.append((x0, y, z))                # riser
    steps = np.array(steps)
    rail = np.array([(x, 0.45, 0.17 * (int((x - 0.5) / 0.25) + 1) + 0.9) for x in xs]
                    + [(x, 0.45, 0.17 * (int((x - 0.5) / 0.25) + 1) + h) for x in xs for h in (0.4, 0.6)])
    under = np.array([(x, 0.0, 2.2) for x in xs])       # underside of an upper flight
    floor = np.array([(x, y, 0.0) for x in np.arange(-0.5, 0.5, 0.05) for y in np.arange(-0.4, 0.41, 0.05)])
    tread_rail = np.array([(x, 0.45, 0.17 * (int((x - 0.5) / 0.25) + 1)) for x in xs])   # step below the rail
    stairs = np.concatenate([steps, rail, under, tread_rail])
    kept = m.trusted_stairs(stairs, floor, z_r, window=0.6, column_gap=0.3)
    assert kept[:, 2].max() <= 0.6                      # only the first steps (window)
    assert not any(np.isclose(kept[:, 1], 0.45) & (kept[:, 2] > 0.17 * 4 - 1e-9))   # no railing / balusters
    near_rail = kept[np.isclose(kept[:, 1], 0.45)]
    assert len(near_rail) and (near_rail[:, 2] <= 0.17 * 3 + 1e-9).all()             # the treads beside it stay
    first = steps[steps[:, 2] <= 0.6]
    assert len(kept[~np.isclose(kept[:, 1], 0.45)]) == len(first)                     # every riser / tread in the window


def test_support_height_follows_floor_and_steps():
    m = VoxelMapper(resolution=0.05, size_xy=4.0, z_above=3.0, robot_radius=0.22)
    k = m.layer(0.0)
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    m.logodds[:, :, k] = m.l_max                                    # floor
    for n in range(4):                                              # steps of 4 layers (0.2 m), 0.3 m deep (map +-2 m)
        sel = (xs >= 0.5 + 0.3 * n) & (xs < 0.8 + 0.3 * n)
        m.logodds[np.ix_(sel, np.arange(m.ny), range(k, k + 4 * (n + 1) + 1))] = m.l_max
    z_floor = m.origin_z + (k + 0.5) * m.res
    assert abs(m.support_height(0.0, 0.0, z_floor + 0.16) - z_floor) < 1e-6          # base 0.16 above floor
    for n in range(4):
        x = 0.65 + 0.3 * n                                          # middle of step n
        z_step = z_floor + 0.2 * (n + 1)
        assert abs(m.support_height(x, 0.0, z_step + 0.05) - z_step) < 1e-6          # base 0.05 above it
        assert abs(m.support_height(x, 0.0, z_step + 0.18) - z_step) < 1e-6
    assert m.support_height(0.0, 0.0, z_floor + 2.5) is None        # nothing below a base in the air


def test_trusted_stairs_keep_an_upper_flight_above_a_lower_one():
    # a turning staircase: the robot stands at z_r = 2.0 on the upper flight, which runs above
    # the lower flight (1.2 m below): its next treads are trusted, the lower flight does not count
    m = VoxelMapper(resolution=0.05, size_xy=4.0, z_above=3.0, robot_radius=0.22)
    xs = np.arange(0.0, 1.0, 0.05)
    lower = np.array([(x, 0.0, 0.8 + 0.17 * int(x / 0.28)) for x in xs])         # flight below
    upper = np.array([(x, 0.0, 2.0 + 0.17 * int(x / 0.28)) for x in xs])         # flight the robot is on
    kept = m.trusted_stairs(np.concatenate([lower, upper]), np.zeros((0, 3)), 2.0)
    assert len(kept) == len(upper[upper[:, 2] <= 2.6])                            # its first steps, all
    assert (kept[:, 2] >= 2.0 - 1e-9).all()


def test_remembered_stairs_are_trusted_when_the_window_reaches_them():
    # the top step / upper floor edge was seen from below (outside the window) and is not seen
    # again from close by (camera blind spot): it becomes trusted when z_r rises to it
    m = VoxelMapper(resolution=0.05, size_xy=4.0, z_above=3.5, robot_radius=0.22)
    k = m.layer(0.0)
    top = np.array([(x, y, 2.9) for x in np.arange(1.0, 1.3, 0.05) for y in np.arange(-0.3, 0.31, 0.05)])
    m.remember_stairs(top)
    assert m.promote_stairs(2.1) == 0                        # 0.8 m above z_r: not yet
    assert not m.trav_s.any()
    n = m.promote_stairs(2.7)                                # one step below it
    assert n == m.stair_seen.sum() == m.stairs.sum() > 0
    assert m.promote_stairs(2.7) == 0                        # once only


def test_stair_distance_counts_only_stairs_near_the_floor_height():
    # on the upper floor the staircase below is close in xy but 2-3 m lower: it must not keep the
    # robot in mode s
    m, k, xs, step_pts, floor_pts = _step_scene()
    m.integrate_semantic(np.concatenate([floor_pts, step_pts]), step_pts)
    assert m.stair_distance(0.0, 0.0, z_r=0.0) < 0.55
    assert m.stair_distance(0.0, 0.0, z_r=3.0) == float('inf')


def test_floor_seen_in_mode_s_stays_in_mode_g_but_not_the_stairs():
    m, k, xs, step_pts, floor_pts = _step_scene()
    m.set_mode('s')
    m.integrate_semantic(np.concatenate([floor_pts, step_pts]), step_pts)
    i_top = int(np.flatnonzero(xs > 0.6)[0])
    j = m.ny // 2
    i_floor = int(np.flatnonzero(xs < -1.0)[-1])
    m.set_mode('g')
    assert m.trav_g[i_floor, j, k]                       # floor from M_s kept
    assert not m.traversable()[i_top, j, k + 4]           # the step is not traversable in g
