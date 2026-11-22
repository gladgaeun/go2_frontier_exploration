# Unit tests for frontier3d.FrontierManager (pure numpy/scipy, no ROS / Habitat).
#   python3 -m pytest trav_exploration/test/test_frontier3d.py

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from trav_exploration.frontier3d import FrontierManager  # noqa: E402
from trav_exploration.voxel_mapper import VoxelMapper  # noqa: E402
from test_voxel_mapper import build, cell, frame  # noqa: E402


def as_sets(clusters):
    return sorted(tuple(np.sort(v).tolist()) for v in clusters)


def test_closed_room_has_no_frontier():
    m = build()
    f = FrontierManager(m)
    f.reset()
    assert not f.clusters
    # the literal reading (neighbour column not in M_trav) puts frontiers along every wall
    literal = FrontierManager(m, boundary='trav', need_unknown=False)
    literal.reset()
    assert sum(c.size for c in literal.clusters.values()) > 50


def test_frontier_at_unexplored_boundary():
    m = VoxelMapper(resolution=0.1, size_xy=10.0)
    f = FrontierManager(m)
    o, pts, hit = frame(0.0, pos=(-2.5, 0.0))           # looking along +x, back wall behind
    m.integrate(o, pts, hit, support=(-2.5, 0.0, 0.0))
    f.update(m.pop_update_box())
    assert f.clusters
    k = m.layer(0.0)
    vox = f.voxels()
    trav = m.traversable().reshape(-1)
    assert trav[vox].all()                                 # frontiers lie in M_trav
    assert (np.unravel_index(vox, m.logodds.shape)[2] == k).all()
    for c in f.clusters.values():
        assert c.size >= f.min_size
        lo, hi = c.aabb
        ijk = (c.centroid - [m.origin_xy, m.origin_xy, m.origin_z]) / m.res
        assert all(lo[i] <= ijk[i] <= hi[i] for i in range(3))


def test_incremental_equals_global():
    m = VoxelMapper(resolution=0.1, size_xy=10.0)
    f = FrontierManager(m)
    poses = [((-2.0, -2.0), 0.3), ((-2.0, -2.0), 1.6), ((0.0, -1.0), 0.0), ((0.0, -1.0), 2.5),
             ((1.5, 0.0), -1.5), ((-1.0, 1.0), math.pi), ((-1.0, 1.0), 0.8), ((2.0, -2.0), 2.0)]
    for (x, y), yaw in poses:
        o, pts, hit = frame(yaw, pos=(x, y))
        m.integrate(o, pts, hit, support=(x, y, 0.0))
        f.update(m.pop_update_box())
        assert as_sets(c.voxels for c in f.clusters.values()) == as_sets(v for v, _, _ in f.extract_all())


def test_headroom():
    m = VoxelMapper(resolution=0.1, size_xy=4.0)
    k = m.layer(0.0)
    m.trav_g[10:20, 10:20, k] = True
    f = FrontierManager(m, min_size=1)
    f.reset()
    edge = (10, 15, k)
    assert np.ravel_multi_index(edge, m.logodds.shape) in f.voxels()
    # a traversable layer within K_h above (e.g. a mezzanine) removes the frontier below it
    m.trav_g[8:22, 8:22, k + 3] = True
    f.reset()
    vox = set(f.voxels().tolist())
    assert np.ravel_multi_index(edge, m.logodds.shape) not in vox
    assert all(np.unravel_index(v, m.logodds.shape)[2] == k + 3 for v in vox)


def test_small_clusters_discarded():
    m = VoxelMapper(resolution=0.1, size_xy=4.0)
    k = m.layer(0.0)
    m.trav_g[10, 10, k] = True                             # a single traversable voxel
    m.trav_g[20:30, 20:30, k] = True
    f = FrontierManager(m, min_size=4)
    f.reset()
    assert len(f.clusters) == 1
    assert np.ravel_multi_index((10, 10, k), m.logodds.shape) not in f.voxels()


def test_unchanged_clusters_kept():
    m = build()
    f = FrontierManager(m)
    o, pts, hit = frame(0.0, pos=(-2.5, 0.0))
    m2 = VoxelMapper(resolution=0.1, size_xy=10.0)
    f2 = FrontierManager(m2)
    m2.integrate(o, pts, hit, support=(-2.5, 0.0, 0.0))
    f2.update(m2.pop_update_box())
    ids = set(f2.clusters)
    removed, added = f2.update(((0, 0, 0), (1, 1, 1)))      # change far away from every cluster
    assert not removed and not added and set(f2.clusters) == ids
    assert f.update(None) == ([], [])
    assert cell(m, 0.0, 0.0)[2] == m.layer(0.0)


def _partly_trusted_stairs(n_trusted=3, walls=True):
    """test_trav3d's staircase (mode s), only its lowest n_trusted steps trusted: the steps above
    are stair-labelled but not in M_s (obstacles), as before VoxelMapper.trusted_stairs reaches them."""
    from test_trav3d import _stairs
    m, k, top = _stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    up = xs >= 0.5 + 0.3 * n_trusted
    pend = m.stairs & up[:, None, None]
    m.stair_seen |= pend
    m.trav_s &= ~pend
    m.stairs &= ~pend
    m.trav_s[up, :, k + 4 * (6 + 1)] = False             # the landing is not seen yet either
    m.logodds[up, :, k + 4 * (6 + 1)] = 0.0
    if not walls:                                         # a railing side: free space seen there
        side = (np.abs(ys) >= 0.6)
        m.logodds[np.ix_(xs > 0.4, side, range(k + 1, m.nz))] = m.l_min
    m.reinflate_all()
    return m, k


def test_semantic_frontier_at_the_last_trusted_step():
    m, k = _partly_trusted_stairs()
    fm = FrontierManager(m, min_size=3)
    fm.reset()
    stair = [c for c in fm.clusters.values() if m.semantic_only_at(c.voxels).mean() >= 0.5]
    assert stair, 'no semantic frontier on the staircase'
    c = max(stair, key=lambda c: c.size)
    z_last = m.origin_z + (k + 4 * 3 + 0.5) * m.res        # tread of the third step
    assert abs(c.centroid[2] - z_last) < 0.15 and 1.0 < c.centroid[0] < 1.6
    for c in stair:                                       # across the top edge, not along the sides
        assert np.ptp(m.centers(c.voxels)[:, 0]) <= 0.15


def test_no_semantic_frontier_along_a_seen_void():
    m, k = _partly_trusted_stairs(walls=False)
    fm = FrontierManager(m, min_size=3)
    fm.reset()
    stair = [c for c in fm.clusters.values() if m.semantic_only_at(c.voxels).mean() >= 0.5]
    assert stair
    for c in stair:                                       # the void beside the steps: no frontier
        assert np.ptp(m.centers(c.voxels)[:, 0]) <= 0.15


def test_no_semantic_frontier_along_a_railing_labelled_stairs():
    # a railing beside the steps, labelled stairs (not trusted: pending), 0.9 m high
    m, k = _partly_trusted_stairs()
    xs = m.origin_xy + (np.arange(m.nx) + 0.5) * m.res
    ys = m.origin_xy + (np.arange(m.ny) + 0.5) * m.res
    rail = np.ix_((xs > 0.4) & (xs < 1.5), (ys >= 0.6) & (ys < 0.7), range(k, k + 4 * 3 + 18))
    m.logodds[rail] = m.l_max
    m.stair_seen[rail] = True
    m.reinflate_all()
    fm = FrontierManager(m, min_size=3)
    fm.reset()
    stair = [c for c in fm.clusters.values() if m.semantic_only_at(c.voxels).mean() >= 0.5]
    assert stair
    for c in stair:
        assert np.ptp(m.centers(c.voxels)[:, 0]) <= 0.15


def _one_frontier():
    m = VoxelMapper(resolution=0.1, size_xy=10.0)
    f = FrontierManager(m)
    o, pts, hit = frame(0.0, pos=(-2.5, 0.0))
    m.integrate(o, pts, hit, support=(-2.5, 0.0, 0.0))
    f.update(m.pop_update_box())
    return m, f


def test_dormant_flag_kept_while_voxels_stay():
    m, f = _one_frontier()
    c = max(f.clusters.values(), key=lambda c: c.size)
    others = [k.id for k in f.clusters.values() if k.id != c.id]
    assert f.mark_visited(c.id) is c and c.visited
    # the region is re-segmented without any change: the same voxels keep the flag
    lo, hi = c.aabb
    f.update((lo, hi))
    assert c.id not in f.clusters
    same = [k for k in f.clusters.values() if np.array_equal(k.voxels, c.voxels)]
    assert same and same[0].visited
    assert f.resolve(c.id) is same[0]                    # an old id still finds its cluster
    assert all(not k.visited for k in f.clusters.values() if k is not same[0])
    assert not any(f.clusters[i].visited for i in others if i in f.clusters)


def test_release_clears_all_dormant_flags_but_keeps_failures():
    m, f = _one_frontier()
    c = max(f.clusters.values(), key=lambda c: c.size)
    f.mark_failed(c.id, 2)
    f.mark_failed(c.id, 2)
    assert c.visited and c.fails == 2
    assert f.release_dormant() == 1
    assert not c.visited and c.anchor is None and c.fails == 2
    assert not any(k.visited for k in f.clusters.values())
    f.mark_failed(c.id, 2)                               # failing again: dormant at once
    assert c.visited


def test_dormant_flag_reset_when_space_changes():
    m, f = _one_frontier()
    c = max(f.clusters.values(), key=lambda c: c.size)
    f.mark_visited(c.id)
    # the robot sees the space beyond that frontier from elsewhere: its voxels move
    for (x, y), yaw in [((0.0, 0.0), 0.0), ((0.0, 0.0), 1.5), ((0.0, 0.0), -1.5)]:
        o, pts, hit = frame(yaw, pos=(x, y))
        m.integrate(o, pts, hit, support=(x, y, 0.0))
        f.update(m.pop_update_box())
    for k in f.clusters.values():                          # compared with the dormant voxels
        assert k.visited == f._same_vox(c.anchor, k.voxels)
    assert any(not k.visited and f._near(k.voxels, c.anchor) > 0 for k in f.clusters.values())


def test_no_frontier_at_the_hidden_floor_along_a_wall():
    # the floor strip at the foot of a wall was never seen (unknown, not T) but lies in O_inf:
    # no pose can stand there, so its edge is no frontier (it was, before need_unknown)
    m = build()
    k = m.layer(0.0)
    i0, _, _ = cell(m, 2.8, 0.0)
    i1, _, _ = cell(m, 3.0, 0.0)
    _, j0, _ = cell(m, 0.0, -2.0)
    _, j1, _ = cell(m, 0.0, 2.0)
    strip = (slice(i0, i1), slice(j0, j1))
    m.trav_g[strip[0], strip[1], :] = False
    m.logodds[strip[0], strip[1], k:k + 3] = 0.0
    m.reinflate_all()
    assert m.inflated[strip[0], strip[1], k].all()       # inside O_inf
    f = FrontierManager(m)
    f.reset()
    assert not f.clusters
    loose = FrontierManager(m, need_unknown=False)
    loose.reset()
    assert loose.clusters                                  # the old rule: a frontier along it


def test_frontier_at_an_unexplored_gap_in_the_floor():
    m = build()
    k = m.layer(0.0)
    i0, j0, _ = cell(m, 0.5, 0.5)
    hole = (slice(i0, i0 + 6), slice(j0, j0 + 6))
    m.trav_g[hole[0], hole[1], k] = False
    m.logodds[hole[0], hole[1], :] = 0.0                   # never seen
    m.occupied[hole[0], hole[1], :] = False
    m.reinflate_all()
    f = FrontierManager(m)
    f.reset()
    assert f.clusters


def test_frontier_where_grazing_rays_marked_the_floor_free():
    # far floor: rays grazing it mark the voxels free without a hit (not T, not unknown); that
    # edge is still a frontier (requiring an unknown voxel dropped it)
    m = build()
    k = m.layer(0.0)
    i0, j0, _ = cell(m, 0.5, 0.5)
    gap = (slice(i0, i0 + 6), slice(j0, j0 + 6))
    m.trav_g[gap[0], gap[1], k] = False
    m.logodds[gap[0], gap[1], :] = m.l_min
    m.occupied[gap[0], gap[1], :] = False
    m.reinflate_all()
    f = FrontierManager(m)
    f.reset()
    assert f.clusters


def test_failed_goals_make_a_cluster_dormant_and_go_with_it():
    m, f = _one_frontier()
    c = max(f.clusters.values(), key=lambda c: c.size)
    assert f.mark_failed(c.id, max_attempts=2) is c and c.fails == 1 and not c.visited
    f.update(c.aabb)                                       # re-segmented, same voxels
    same = f.resolve(c.id)
    assert same is not None and same.fails == 1           # the count went with it
    f.mark_failed(c.id, max_attempts=2)                    # the old id still finds it
    assert same.fails == 2 and same.visited


def test_failure_count_reset_when_space_changes():
    m, f = _one_frontier()
    c = max(f.clusters.values(), key=lambda c: c.size)
    f.mark_failed(c.id, max_attempts=3)
    for (x, y), yaw in [((0.0, 0.0), 0.0), ((0.0, 0.0), 1.5), ((0.0, 0.0), -1.5)]:
        o, pts, hit = frame(yaw, pos=(x, y))
        m.integrate(o, pts, hit, support=(x, y, 0.0))
        f.update(m.pop_update_box())
    old = set(c.voxels.tolist())
    for k in f.clusters.values():
        if len(old & set(k.voxels.tolist())) < f.keep_share * k.size:
            assert k.fails == 0


def test_dormant_flag_kept_when_the_frontier_shifts_a_voxel():
    m, f = _one_frontier()
    c = max(f.clusters.values(), key=lambda c: c.size)
    f.mark_visited(c.id)
    nx, ny, nz = m.logodds.shape
    shifted = np.unique(np.concatenate([c.voxels, c.voxels[:5] + ny * nz]))   # a few more at its edge
    assert f._same_vox(c.anchor, shifted)
