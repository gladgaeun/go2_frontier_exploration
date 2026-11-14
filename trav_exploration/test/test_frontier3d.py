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
    literal = FrontierManager(m, boundary='trav')
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
