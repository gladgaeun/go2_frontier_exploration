# Unit tests for voxel_mapper.VoxelMapper on a synthetic room (pure numpy, no ROS / Habitat).
#   python3 -m pytest trav_exploration/test/test_voxel_mapper.py

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.voxel_mapper import VoxelMapper  # noqa: E402

# 6 x 6 m room (walls 0.2 m thick, 2.5 m high), floor at z = 0, a table with its top at
# 0.3-0.35 m with free space below (the robot, 0.6 m high, cannot pass under it) and a low box (0.3 m) on the floor.
BOXES = [
    ((-3.2, -3.2, -0.1), (3.2, 3.2, 0.0)),     # floor
    ((-3.2, -3.2, 0.0), (-3.0, 3.2, 2.5)),     # walls
    ((3.0, -3.2, 0.0), (3.2, 3.2, 2.5)),
    ((-3.2, -3.2, 0.0), (3.2, -3.0, 2.5)),
    ((-3.2, 3.0, 0.0), (3.2, 3.2, 2.5)),
    ((1.0, 1.0, 0.3), (2.0, 2.0, 0.35)),       # table top (overhang)
    ((-2.0, 1.0, 0.0), (-1.5, 1.5, 0.3)),      # low box
]
CAM_H = 0.5


def raycast(o, dirs, max_range):
    """Distance to the first box hit along every direction (inf if none within max_range)."""
    best = np.full(len(dirs), np.inf)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    for lo, hi in BOXES:
        t1 = (np.array(lo) - o) * inv
        t2 = (np.array(hi) - o) * inv
        tn = np.minimum(t1, t2).max(axis=1)
        tf = np.maximum(t1, t2).min(axis=1)
        ok = (tf >= tn) & (tn > 1e-6)
        best = np.where(ok & (tn < best), tn, best)
    return np.where(best <= max_range, best, np.inf)


def frame(yaw, pos=(0.0, 0.0), w=80, h=60, max_range=5.0):
    """Points of one pinhole depth frame (hfov 90 deg) looking along yaw."""
    f = (w / 2.0) / math.tan(math.radians(45.0))
    u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2.0, np.arange(h) + 0.5 - h / 2.0)
    # camera: forward = yaw, left = yaw + 90 deg, up = z
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    d = fwd[None, :] - (u.reshape(-1, 1) / f) * left[None, :] - (v.reshape(-1, 1) / f) * up[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.array([pos[0], pos[1], CAM_H])
    r = raycast(o, d, max_range)
    hit = np.isfinite(r)
    pts = o + d * np.where(hit, r, max_range)[:, None]
    return o, pts, hit


def build(yaws=np.linspace(-math.pi, math.pi, 16, endpoint=False), **kw):
    m = VoxelMapper(resolution=0.1, size_xy=10.0, z_below=0.5, z_above=3.0, **kw)
    for yaw in yaws:
        o, pts, hit = frame(yaw)
        m.integrate(o, pts, hit, support=(0.0, 0.0, 0.0))
    return m


def cell(m, x, y):
    k = m.layer(0.0)
    ix = int(np.floor((x - m.origin_xy) / m.res))
    iy = int(np.floor((y - m.origin_xy) / m.res))
    return ix, iy, k


def test_incremental_inflation_equals_global():
    m = build()
    inc = m.inflated.copy()
    m.reinflate_all()
    assert np.array_equal(inc, m.inflated)


def test_traversability_layers():
    m = build()
    trav = m.traversable()
    assert trav[cell(m, 0.0, 0.0)]                 # under the robot
    assert trav[cell(m, -1.0, -1.5)]               # open floor
    assert not trav[cell(m, 2.9, 0.0)]             # within robot_radius of the wall
    assert m.inflated[cell(m, 2.9, 0.0)]
    assert not trav[cell(m, 1.5, 1.5)]             # under the table top (lower than the robot)
    assert m.inflated[cell(m, 1.5, 1.5)]
    assert m.inflated[cell(m, -1.75, 1.25)]        # low box on the floor
    # floor voxels are occupied but must not inflate their own layer
    assert m.occupied[cell(m, -1.0, -1.5)]
    # ceiling-free: nothing above the robot height inflates the support layer
    assert np.array_equal(m.inflated[:, :, m.layer(0.0)],
                          m._dilate(m.occupied & ~m.trav_g)[:, :, m.layer(0.0)])


def test_trav_grid_is_closed_by_walls():
    m = build()
    data, w, h, ox, oy = m.trav_grid(0.0)
    g = data.reshape(h, w)
    free = g == VoxelMapper.FREE
    unknown = g == VoxelMapper.UNKNOWN
    # free space never touches unknown space beyond the walls (no frontier through walls)
    xs = ox + (np.arange(w) + 0.5) * m.res
    ys = oy + (np.arange(h) + 0.5) * m.res
    outside = (np.abs(xs)[None, :] > 3.0) | (np.abs(ys)[:, None] > 3.0)
    assert not (free & outside).any()
    assert free.sum() * m.res ** 2 > 15.0            # most of the 36 m^2 room minus margins
    assert unknown.any()


def test_trav_grid_takes_a_floor_split_over_two_layers():
    # a floor that lies one layer below z_r over part of the room (HM3D floors are not on one
    # voxel layer) is still free on the Nav2 grid
    m = build()
    k = m.layer(0.0)
    half = slice(0, m.nx // 2)
    m.trav_g[half, :, k - 1] = m.trav_g[half, :, k]
    m.trav_g[half, :, k] = False
    m.reinflate_all()
    data, w, h, ox, oy = m.trav_grid(0.0)
    g = data.reshape(h, w)
    x = ox + (np.arange(w) + 0.5) * m.res
    left = (x < m.origin_xy + (m.nx // 2) * m.res)[None, :] & (g == VoxelMapper.FREE)
    assert left.sum() * m.res ** 2 > 5.0             # the lowered half is on the grid
    m.grid_layers = 0
    data0, w0, h0, ox0, _ = m.trav_grid(0.0)
    g0 = data0.reshape(h0, w0)
    x0 = ox0 + (np.arange(w0) + 0.5) * m.res
    assert not ((x0 < m.origin_xy + (m.nx // 2) * m.res)[None, :] & (g0 == VoxelMapper.FREE)).any()


def test_mode_switch_semantic():
    m = build()
    before = m.traversable().copy()
    m.set_mode('s')
    assert (m.traversable() | ~before).all()     # mode s keeps M_g (T = M_g U M_s): no holes
    floor = np.stack(np.meshgrid(np.arange(-2.5, 2.5, 0.05), np.arange(-2.5, 2.5, 0.05)), -1).reshape(-1, 2)
    m.integrate_semantic(np.column_stack([floor, np.full(len(floor), -0.01)]))
    inc = m.inflated.copy()
    m.reinflate_all()
    assert np.array_equal(inc, m.inflated)
    assert m.traversable()[cell(m, -1.0, -1.5)]
    m.set_mode('g')
    assert m.traversable()[cell(m, -1.0, -1.5)]


def test_geometric_ray_casting():
    m = build()
    g = m.trav_g
    assert g[cell(m, 0.0, 0.0)]                    # robot support state x_r
    assert g[cell(m, -1.0, -1.5)]                  # floor between the robot and the walls
    # occupied voxels are skipped: an observed obstacle standing on the support voxel
    assert m.occupied[cell(m, 3.05, 0.05)[:2] + (m.layer(0.0) + 1,)]
    assert not g[cell(m, 3.05, 0.05)]              # wall
    # the inside of the low box is never observed (only its top), so a ray marks it, and the
    # box top removes it from M_trav through the downward inflation
    assert not m.traversable()[cell(m, -1.75, 1.25)]
    # an overhang leaves the support voxel free: in M_g, but removed from M_trav by O_inf
    assert g[cell(m, 1.45, 1.45)]
    assert not m.traversable()[cell(m, 1.45, 1.45)]
    # M_g lies on the support layer only
    k = m.layer(0.0)
    assert not g[:, :, :k].any() and not g[:, :, k + 1:].any()


def test_geometric_needs_obstacle_points():
    # rays start at obstacle points only: open floor with no obstacle within range stays out of M_g
    m = VoxelMapper(resolution=0.1, size_xy=10.0)
    o, pts, hit = frame(0.0, pos=(-2.5, 0.0), max_range=2.0)
    m.integrate(o, pts, hit, support=(-2.5, 0.0, 0.0))
    assert m.occupied[cell(m, -1.25, 0.05)]        # floor seen there ...
    assert not m.trav_g[cell(m, -1.25, 0.05)]      # ... but no ray passes over it


def test_geometric_ray_casting_in_mode_s_on_request():
    # mode s keeps M_g as it is unless the frame asks for it (the node does on a floor)
    m = VoxelMapper(resolution=0.1, size_xy=10.0, z_below=0.5, z_above=3.0, mode='s')
    o, pts, hit = frame(0.0)
    m.integrate(o, pts, hit, support=(0.0, 0.0, 0.0))
    assert not m.trav_g.any()
    m.integrate(o, pts, hit, support=(0.0, 0.0, 0.0), update_g=True)
    assert m.trav_g[cell(m, 1.5, 0.0)]


def test_geometric_leaves_out_a_seen_void():
    # a 1 x 1 m stairwell in the floor (lower floor 0.45 m down): the 2-D rays from the far wall
    # pass over it, but the camera saw through it at floor height, so M_g stays off it
    global BOXES
    saved = BOXES
    BOXES = [((-3.2, -3.2, -0.1), (1.5, 3.2, 0.0)),        # floor up to the hole
             ((2.5, -3.2, -0.1), (3.2, 3.2, 0.0)),         # and after it
             ((1.5, -3.2, -0.1), (2.5, -1.0, 0.0)),
             ((1.5, 0.0, -0.1), (2.5, 3.2, 0.0)),
             ((1.5, -1.0, -0.55), (2.5, 0.0, -0.45))] + saved[1:5]
    try:
        m = build()
        hole = cell(m, 2.05, -0.45)
        assert not m.trav_g[hole]
        assert m.trav_g[cell(m, 1.05, -0.45)] and m.trav_g[cell(m, 2.85, -0.45)]
        # M_g voxels marked there before the camera saw through are dropped
        m.trav_g[hole] = True
        o, pts, hit = frame(-0.2)
        m.integrate(o, pts, hit, support=(0.0, 0.0, 0.0))
        assert not m.trav_g[hole]
    finally:
        BOXES = saved


def test_geometric_only_grows():
    m = build()
    before = m.trav_g.copy()
    for yaw in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        o, pts, hit = frame(yaw, pos=(-1.0, -1.0))
        m.integrate(o, pts, hit, support=(-1.0, -1.0, 0.0))
    assert (m.trav_g | before).sum() == m.trav_g.sum()


def test_mark_blocked_closes_a_gap():
    m = build()
    target = cell(m, 1.05, -0.95)                      # voxel centres, not boundaries
    assert m.traversable()[target]
    assert m.mark_blocked([(1.05, -0.95)], 0.0) == 1
    assert not m.traversable()[target]
    assert not m.traversable()[cell(m, 1.05, -1.15)]   # within robot_radius of v_front
    # later frames see free space through it (a mesh hole), the block must stay
    for yaw in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        o, pts, hit = frame(yaw)
        m.integrate(o, pts, hit, support=(0.0, 0.0, 0.0))
    assert not m.traversable()[target]
    assert len(m.blocked_points()) == 1
    inc = m.inflated.copy()
    m.reinflate_all()
    assert np.array_equal(inc, m.inflated)


def test_stuck_detector():
    from trav_exploration.voxel_mapper import StuckDetector
    dt, v = 0.05, 0.4
    free, blocked = StuckDetector(), StuckDetector()
    x = 0.0
    hits = []
    for i in range(1, 60):
        t = i * dt
        x += v * dt
        assert not free.update(t, x, 0.0, 0.0, v, 0.0)        # moves as commanded
        # blocked at x = 0.3: the command keeps pushing forward
        hits.append(blocked.update(t, min(x, 0.3), 0.0, 0.0, v, 0.0))
    assert any(hits)
    assert not any(hits[:int(0.3 / (v * dt))])           # not before the wall is reached
    slow = StuckDetector()                               # Nav2 creeping near obstacles (0.07 m/s)
    assert any(slow.update(i * dt, min(0.07 * i * dt, 0.02), 0.0, 0.0, 0.07, 0.0) for i in range(1, 60))
    creep = StuckDetector()
    assert not any(creep.update(i * dt, 0.07 * (i - 2) * dt, 0.0, 0.0, 0.07, 0.0) for i in range(1, 60))
    rot = StuckDetector()                                # turning in place is not a forward action
    assert not any(rot.update(i * dt, 0.0, 0.0, 0.5 * i * dt, 0.0, 0.5) for i in range(1, 60))


def test_traverse_2d_matches_dense_sampling():
    from trav_exploration.voxel_mapper import traverse_2d
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 100001)[:, None]
    for _ in range(100):
        a, b = rng.uniform(0, 20, 2), rng.uniform(0, 20, 2)
        a = np.floor(a) + 0.5                               # rays start at cell centres
        ref = set(map(tuple, np.floor(a + (b - a) * t).astype(int)))
        assert set(map(tuple, traverse_2d(a, b))) == ref


def test_threshold_below_h_min_is_passable():
    # a 6 cm door threshold across the room: occupied, but below h_min (0.1 m), so it neither
    # blocks M_g nor inflates; the robot can cross it
    BOXES.append(((0.95, -3.0, 0.0), (1.05, 3.0, 0.06)))
    try:
        m = build()
    finally:
        BOXES.pop()
    k = m.layer(0.0)
    on = cell(m, 1.0, -1.5)
    assert m.occupied[on[0], on[1], k + 1]                    # the threshold is in the map
    for x in (0.75, 1.0, 1.25):                               # before, on and behind it
        assert m.traversable()[cell(m, x, -1.5)]
    assert not m.traversable()[cell(m, -1.75, 1.25)]          # the 0.3 m box still blocks


def test_traversable_at_matches_traversable_in_both_modes():
    m = build()
    full = m.traversable().reshape(-1)
    idx = np.concatenate([np.flatnonzero(full)[:2000], np.random.default_rng(0).integers(0, full.size, 2000)])
    assert np.array_equal(m.traversable_at(idx), full[idx])
    m.trav_s[:, :, m.layer(0.0) + 3] = True                # something only in M_s
    m.set_mode('s')
    full = m.traversable().reshape(-1)
    idx = np.concatenate([np.flatnonzero(full)[-2000:], idx])
    assert np.array_equal(m.traversable_at(idx), full[idx])


def test_semantic_only_is_stairs_outside_m_g():
    m = VoxelMapper(resolution=0.1, size_xy=4.0, z_below=0.5, z_above=2.0, mode='s')
    k = m.layer(0.0)
    m.trav_s[10:20, 10:20, k] = True                       # floor seen in mode s
    m.trav_s[20:25, 10:20, k + 2] = True                   # a step
    m.stairs[20:25, 10:20, k + 2] = True
    flat = lambda i, j, kk: (i * m.ny + j) * m.nz + kk
    assert not m.semantic_only_at([flat(12, 12, k)])[0]    # floor: joins M_g at s -> g
    assert m.semantic_only_at([flat(22, 12, k + 2)])[0]    # the step: only through M_s
    m.trav_g[22, 12, k + 2] = True
    assert not m.semantic_only_at([flat(22, 12, k + 2)])[0]


def test_virtual_ground_counts_points_below_the_plane_near_the_robot():
    from trav_exploration.voxel_mapper import VirtualGround
    vg = VirtualGround(depth=0.3, radius=2.0, n_down=2, hold=3.0)
    pts = [[1.0, 0.0, -0.1],             # floor noise / one step down: above Pi_vg
           [1.0, 0.0, -0.5], [1.5, 0.0, -1.0], [0.0, 1.0, -2.9],   # stairs and the floor below
           [3.0, 0.0, -3.0]]             # far in xy: not under the robot
    assert vg.count(pts, 0.0, 0.0, 0.0) == 3
    assert vg.count(pts, 0.0, 0.0, -3.0) == 0          # standing on the lower floor


def test_virtual_ground_hysteresis():
    from trav_exploration.voxel_mapper import VirtualGround
    vg = VirtualGround(n_down=10, hold=3.0)
    assert vg.update(5, 0.0) == (False, False)
    assert vg.update(11, 1.0) == (True, True)          # more than N_down: look down
    assert vg.update(0, 2.0) == (True, False)          # held
    assert vg.update(11, 3.5) == (True, False)
    assert vg.update(0, 6.0) == (True, False)
    assert vg.update(0, 6.5) == (False, True)          # hold s without points below: look up


def test_floor_area_counts_floor_not_stairs():
    m = VoxelMapper(resolution=0.05, size_xy=8.0, z_above=3.0, robot_radius=0.22, mode='s')
    k = m.layer(0.0)
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    m.trav_s[xs < 0.0, :, k] = True                       # floor on x < 0
    m.trav_s[(xs >= 0.0) & (xs < 1.0), :, k] = True       # stairs at the same height on 0 <= x < 1
    m.stairs[(xs >= 0.0) & (xs < 1.0), :, k] = True
    full = math.pi * 2.5 ** 2
    a = m.floor_area(0.0, 0.0, 0.0)
    assert 0.4 * full < a < 0.6 * full                    # the floor half of the disc only
    assert m.floor_area(0.0, 0.0, 1.0) == 0.0             # nothing at another height
    assert m.floor_area(3.0, 0.0, 0.0) < 0.1 * full       # beside the stairs, away from the floor


def test_start_support_rejects_a_floor_far_under_the_robot():
    from trav_exploration.voxel_mapper import start_support
    # on a stair landing: the floor below seen down the stairwell, 0.58 m under base_link
    z, rejected = start_support(-0.579, 0.0, 0.592, 0.88)
    assert rejected and abs(z - (0.592 - 0.88)) < 1e-9
    # on a floor: base_link 0.1 m above it
    assert start_support(-0.1, 0.0, 0.79, 0.88) == (-0.1, False)
    assert start_support(None, 0.0, 0.79, 0.88) == (None, False)


def test_virtual_ground_looks_down_until_it_sees_no_more_below():
    """Closed loop: one 30 deg step on the trigger, a further step while it shows clearly more
    below Pi_vg, held where it stops paying; level again after hold s without points below."""
    import math
    from trav_exploration.voxel_mapper import VirtualGround
    vg = VirtualGround(n_down=10, hold=3.0, step=math.radians(30), max_pitch=math.radians(60),
                       settle=2, gain=0.1, rescan_move=1.0)
    assert vg.update(11, 0.0, (0, 0)) == (True, True)
    assert math.isclose(vg.pitch, math.radians(30))
    vg.update(50, 0.1, (0, 0)); vg.update(50, 0.2, (0, 0))     # settling: frames of the old pitch
    assert math.isclose(vg.pitch, math.radians(30))
    vg.update(50, 0.3, (0, 0))                                 # 50 > 1.1 x 11: one step further
    assert math.isclose(vg.pitch, math.radians(60)) and not vg.settled
    for t in (0.4, 0.5):
        vg.update(30, t, (0, 0))
    vg.update(30, 0.6, (0, 0))                                 # fewer than at 30 deg: back, held
    assert vg.settled and math.isclose(vg.pitch, math.radians(30))
    vg.update(30, 0.7, (0.5, 0))
    assert vg.settled                                          # not moved far enough to search again
    vg.update(30, 0.8, (1.2, 0))                               # moved 1.2 m: search one step further
    assert not vg.settled and math.isclose(vg.pitch, math.radians(60))
    assert vg.update(0, 4.0, (1.2, 0)) == (False, True) and vg.pitch == 0.0


def test_virtual_ground_holds_when_looking_further_down_adds_nothing():
    import math
    from trav_exploration.voxel_mapper import VirtualGround
    vg = VirtualGround(n_down=10, step=math.radians(30), max_pitch=math.radians(60), settle=0)
    vg.update(100, 0.0)
    vg.update(105, 0.1)                                        # < 1.1 x 100: not worth going on
    assert vg.settled and math.isclose(vg.pitch, math.radians(30))


def test_virtual_ground_forced_going_down_stairs():
    """Going down stairs: at least one step down without the trigger and no release."""
    import math
    from trav_exploration.voxel_mapper import VirtualGround
    vg = VirtualGround(n_down=10, hold=3.0, step=math.radians(30), max_pitch=math.radians(60), settle=0)
    vg.force(0.0)
    assert vg.active and math.isclose(vg.pitch, math.radians(30))
    vg.update(5, 0.1)                                          # first measure: one step further
    assert math.isclose(vg.pitch, math.radians(60))
    vg.update(2, 0.2)                                          # fewer: back to one step, not level
    assert vg.settled and math.isclose(vg.pitch, math.radians(30))
    assert vg.update(0, 10.0) == (True, False)                 # no release while forced
    vg.reset()
    assert vg.pitch == 0.0


def _solid(m, x0, x1, y0, y1, z_top):
    """Occupied box from the floor (z 0) up to z_top over x0..x1, y0..y1 (a step, a sofa...)."""
    i0, i1 = (int((v - m.origin_xy) / m.res) for v in (x0, x1))
    j0, j1 = (int((v - m.origin_xy) / m.res) for v in (y0, y1))
    m.logodds[i0:i1, j0:j1, m.layer(-0.05):m.layer(z_top - 1e-6) + 1] = m.l_occ + 1.0


def _floor(m):
    m.logodds[:, :, m.layer(-0.05)] = m.l_occ + 1.0


def test_stair_steps_counts_the_risers_of_a_flight():
    m = VoxelMapper(resolution=0.05, size_xy=6.0, z_above=2.0)
    _floor(m)
    for s in range(1, 6):                      # five 0.18 m steps, 0.3 m treads, 1 m wide
        _solid(m, 0.3 * s, 0.3 * s + 0.3, -0.5, 0.5, 0.18 * s)
    risers, n_high, levels = m.stair_steps((0.0, 0.0, 0.0), (1.8, 0.0, 0.9))
    assert risers >= 4 and n_high > 400 and len(levels) >= 5


def test_stair_steps_a_threshold_or_a_sofa_is_no_flight():
    m = VoxelMapper(resolution=0.05, size_xy=6.0, z_above=2.0)
    _floor(m)
    _solid(m, 0.5, 0.7, -0.5, 0.5, 0.15)       # a 0.15 m threshold
    risers, n_high, _ = m.stair_steps((0.0, 0.0, 0.0), (1.5, 0.0, 0.15))
    assert risers <= 1 and n_high >= 60        # raised part seen: no flight
    m0 = VoxelMapper(resolution=0.05, size_xy=6.0, z_above=2.0)
    _floor(m0)                                 # only the floor seen so far (stairs far away)
    assert m0.stair_steps((0.0, 0.0, 0.0), (1.8, 0.0, 0.9))[1] == 0   # nothing raised observed: no verdict
    m2 = VoxelMapper(resolution=0.05, size_xy=6.0, z_above=2.0)
    _floor(m2)
    _solid(m2, 0.5, 1.3, -1.0, 1.0, 0.45)      # sofa seat
    _solid(m2, 1.3, 1.6, -1.0, 1.0, 0.85)      # its back
    risers, n_high, _ = m2.stair_steps((0.0, 0.0, 0.0), (1.6, 0.0, 0.85))
    assert risers == 0 and n_high > 300        # observed well, no step chain
