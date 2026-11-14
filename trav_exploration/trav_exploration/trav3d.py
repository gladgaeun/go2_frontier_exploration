# 3-D path search on the traversable support voxels (TravExplorer Sec. VI-B).
#
# The search runs on M_trav (support voxels, inflated obstacles removed), not on the free 3-D
# space. Two voxels are connected when they lie within `reach` cells in xy and `step_layers`
# layers in z: on a floor that is the 8-neighbourhood, on stairs a step up or down. Due to the
# limited camera FOV the traversable voxels of a staircase can be locally disconnected; the
# foothold-scale reach of 2 cells (0.1 m) bridges such small gaps (the paper's "foothold-scale
# neighborhoods ... preserve connectivity across stairs and floors").
# Transition cost (eq. 15-16): J_step(v_i, v_j) = r |dk_ij| + lambda_esdf * Phi(v_j),
#   Phi(v) = max(0, (d_safe - d_obs(v)) / d_safe), d_obs = distance to the nearest obstacle.
#
# SupportGraph: one graph of the whole map (every floor, and in mode s the stairs between them)
# for the paths on a floor, J_geo of the frontier tour and the stair climb (SupportGraph.climb:
# the highest reachable node near the stair voxels; the trusted part of a staircase grows as z_r
# rises, VoxelMapper.trusted_stairs, so a robot that goes there, re-plans and goes on climbs the
# stairs stretch by stretch).
#
# Pure numpy/scipy, no ROS.

import math

import numpy as np


def graph_from_arrays(trav, obst, res, step_layers=5, reach=2, lambda_esdf=0.1, d_safe=0.5, d_obs=None):
    """support_graph on given arrays: trav = M_trav voxels (nodes), obst = obstacle voxels (d_obs),
    or d_obs itself (same shape as trav).

    Built from the node list (one gather per neighbour offset), not by shifting the whole box: the
    graph of the whole map (SupportGraph) spans every floor and the stairs between them, a box of
    millions of voxels with a few percent of them nodes."""
    from scipy import ndimage, sparse
    if d_obs is None:
        d_obs = ndimage.distance_transform_edt(~obst) * res if obst.any() else np.full(trav.shape, np.inf)
    phi = np.maximum(0.0, (d_safe - d_obs) / d_safe) if d_safe > 0 else np.zeros(trav.shape)
    sx, sy, sz = trav.shape
    pos = np.argwhere(trav)                            # row-major, the order of index[trav]
    n = len(pos)
    index = np.full(trav.shape, -1, dtype=np.int64)
    index[trav] = np.arange(n)
    if n == 0:
        return sparse.csr_matrix((0, 0)), index, trav
    phi_n = phi[trav]
    dz = min(step_layers, sz - 1)
    pad = np.array([reach, reach, dz])
    ipad = np.full((sx + 2 * reach, sy + 2 * reach, sz + 2 * dz), -1, dtype=np.int64)
    ipad[reach:reach + sx, reach:reach + sy, dz:dz + sz] = index
    strides = np.array([ipad.shape[1] * ipad.shape[2], ipad.shape[2], 1])
    flat = (pos + pad) @ strides
    lin = ipad.reshape(-1)
    nodes = np.arange(n)
    rows, cols, vals = [], [], []
    offsets = [(di, dj) for di in range(-reach, reach + 1) for dj in range(-reach, reach + 1)
               if (di or dj) and di * di + dj * dj <= reach * reach]
    for di, dj in offsets:
        for dk in range(-dz, dz + 1):
            nb = lin[flat + (di * strides[0] + dj * strides[1] + dk)]
            ok = nb >= 0
            if not ok.any():
                continue
            nb = nb[ok]
            rows.append(nodes[ok])
            cols.append(nb)
            vals.append(res * math.sqrt(di * di + dj * dj + dk * dk) + lambda_esdf * phi_n[nb])
    if not rows:
        return sparse.csr_matrix((n, n)), index, trav
    g = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return g, index, trav


def _nearest(index, lo, m, p, radius, dz=None):
    """Node of the traversable voxel nearest to point p within radius (3-D), or -1; with dz, only
    voxels within dz of p in z (the graph holds every floor: within 0.6 m of a robot on the floor
    lie bed tops and stair steps too)."""
    ijk = np.floor((np.asarray(p, dtype=np.float64) - [m.origin_xy, m.origin_xy, m.origin_z]) / m.res).astype(int) - lo
    r = int(math.ceil(radius / m.res))
    rz = r if dz is None else min(r, int(math.ceil(dz / m.res)))
    win_lo = np.maximum(ijk - [r, r, rz], 0)
    win_hi = np.minimum(ijk + [r + 1, r + 1, rz + 1], index.shape)
    if (win_lo >= win_hi).any():
        return -1
    win = index[tuple(slice(a, b) for a, b in zip(win_lo, win_hi))]
    cand = np.argwhere(win >= 0)
    if not len(cand):
        return -1
    d2 = (((cand + win_lo) - ijk) ** 2).sum(axis=1)
    k = int(np.argmin(d2))
    if d2[k] > r * r:
        return -1
    return int(win[tuple(cand[k])])


def _backtrack(pred, start, goal):
    path = [goal]
    while path[-1] != start:
        p = pred[path[-1]]
        if p < 0:
            return None
        path.append(int(p))
    return path[::-1]


class SupportGraph:
    """The support graph of the whole map (Sec. VI-B): every M_trav voxel is a node, on every
    floor and, in mode s, on the stairs between them (in mode g the stairs are obstacles, eq. 1,
    so the floors are not connected). One graph for the paths Nav2 follows on a floor, for J_geo
    of the frontier tour and for the stair climb, as in the paper ("connectivity across stairs
    and floors"). Built from a snapshot of the map (SupportGraph.snapshot, taken on the map
    thread), so the graph can be built and searched on another thread.

    Changes to M_trav (not in the paper): the voxels next to O_inf (4-neighbours, also in the
    layers right above / below) are not nodes, a one-cell margin for Nav2's controller; the xy
    cells the robot centre has passed through are nodes on their layer (the robot was there:
    O_inf grown later from close-by must not cut its own way); the cells marked too narrow for
    the body (mark_tight) are not."""

    def __init__(self, snap, step_layers=5, reach=2, lambda_esdf=0.1, d_safe=0.5):
        self.res, self.origin, self.lo = snap['res'], snap['origin'], snap['lo']
        self.step_layers = step_layers
        trav, obst = snap['trav'], snap['obst']
        self.graph, self.index, self.trav = graph_from_arrays(trav, obst, self.res, step_layers, reach,
                                                              lambda_esdf, d_safe, d_obs=snap.get('d_obs'))
        self.stairs = snap.get('stairs')
        self.n = int(self.trav.sum())
        self._pos = None
        self._near_stairs = None
        self._dist = {}               # source node -> (dist, pred)

    @staticmethod
    def snapshot(m, visited=None, tight=None, margin=0.5, d_max=1.0):
        """Copy of the arrays the graph needs (call on the thread that writes the map).
        visited / tight: {layer: (nx, ny) bool} (VoxelMapper.visited_layers / tight_layers).
        d_obs is exact up to d_max (>= d_safe: beyond it Phi = 0, eq. 16)."""
        from scipy import ndimage
        src_all = m._source()
        cols = src_all.any(axis=2)
        if visited:
            for cells in visited.values():
                cols = cols | cells
        if not cols.any():
            return None
        ix, iy = np.nonzero(cols)
        mg = int(math.ceil(margin / m.res))
        x0, x1 = max(ix.min() - mg, 0), min(ix.max() + mg + 1, m.nx)
        y0, y1 = max(iy.min() - mg, 0), min(iy.max() + mg + 1, m.ny)
        bxy = (slice(x0, x1), slice(y0, y1))
        src = src_all[bxy]
        inf = m.inflated[bxy]
        nz = src.shape[2]
        # O_inf of a layer and the ones right above / below (as trav_grid), grown by one cell
        # (4-neighbours): Nav2's controller treats a path cell next to O_inf as a collision (its
        # footprint touches it) and kept aborting on paths along the edge of O_inf
        band = inf.copy()
        band[:, :, 1:] |= inf[:, :, :-1]
        band[:, :, :-1] |= inf[:, :, 1:]
        cross = np.zeros((3, 3, 1), dtype=bool)
        cross[1, :, 0] = cross[:, 1, 0] = True
        trav = src & ~ndimage.binary_dilation(band, structure=cross)
        tol = m.layer_tol
        tight_k = {}
        for k, cells in (tight or {}).items():
            c = cells[bxy]
            if c.any():
                trav[c, max(k - tol, 0):k + tol + 1] = False
                for kk in range(max(k - 1, 0), min(k + 2, nz)):
                    tight_k[kk] = tight_k.get(kk, False) | c
        for k, cells in (visited or {}).items():
            if not 0 <= k < nz:
                continue
            near = trav[:, :, max(k - tol, 0):k + tol + 1].any(axis=2)
            add = cells[bxy] & ~near               # passed, but no support voxel left there
            trav[add, k] = True
        obst = ((m.logodds[bxy] >= m.l_occ) | m.blocked[bxy]) & ~src
        # d_obs (eq. 16) as Nav2 sees the obstacles, per layer: distance (xy) to the O_inf of the
        # layers within one of it (as trav_grid) or a cell marked too narrow, plus the O_inf
        # radius, i.e. the distance to the obstacle that O_inf grows from. The 3-D distance to the
        # occupied voxels missed most of the walls and furniture at the robot's height, and the
        # paths ran along the edge of O_inf, where Nav2's controller saw a collision ahead.
        # Per layer only around its nodes (+ d_max): a stair layer holds a step, not a floor.
        d_obs = np.full(trav.shape, np.inf, dtype=np.float32)
        w = int(math.ceil(d_max / m.res)) + 1
        sx, sy = trav.shape[:2]
        for k in np.flatnonzero(trav.any(axis=(0, 1))):
            ii, jj = np.nonzero(trav[:, :, k])
            a0, a1 = max(ii.min() - w, 0), min(ii.max() + w + 1, sx)
            b0, b1 = max(jj.min() - w, 0), min(jj.max() + w + 1, sy)
            lethal = inf[a0:a1, b0:b1, max(k - 1, 0):k + 2].any(axis=2)
            if k in tight_k:
                lethal = lethal | tight_k[k][a0:a1, b0:b1]
            if lethal.any():
                d_obs[a0:a1, b0:b1, k] = ndimage.distance_transform_edt(~lethal) * m.res + m.robot_radius
        return {'res': m.res, 'origin': np.array([m.origin_xy, m.origin_xy, m.origin_z]),
                'lo': np.array([x0, y0, 0]), 'trav': trav, 'obst': obst, 'd_obs': d_obs,
                'stairs': m.stairs[bxy].copy(), 'mode': m.mode}

    def node(self, p, radius, dz=0.15):
        """Node of the M_trav voxel nearest to world point p within radius (3-D) and dz (z), or
        -1. Points are given at the height of their floor (floor_z) or support (z_r)."""
        m = type('Grid', (), {'origin_xy': self.origin[0], 'origin_z': self.origin[2], 'res': self.res})
        return _nearest(self.index, self.lo, m, p, radius, dz)

    def positions(self):
        """World centres of all nodes (N, 3), indexed by node."""
        if self._pos is None:
            pos = np.argwhere(self.index >= 0)
            out = np.empty((self.n, 3))
            out[self.index[tuple(pos.T)]] = (pos + self.lo + 0.5) * self.res + self.origin
            self._pos = out
        return self._pos

    def _search(self, s):
        from scipy.sparse.csgraph import dijkstra
        if s not in self._dist:
            self._dist[s] = dijkstra(self.graph, directed=True, indices=s, return_predecessors=True)
        return self._dist[s]

    def costs(self, sources, targets, source_radius=0.6, target_radius=0.5):
        """J_geo [len(sources), len(targets)] between world points (np.inf: no path / off M_trav)."""
        from scipy.sparse.csgraph import dijkstra
        out = np.full((len(sources), len(targets)), np.inf)
        if self.n == 0:
            return out
        s = np.array([self.node(p, source_radius) for p in sources], dtype=np.int64)
        t = np.array([self.node(p, target_radius) for p in targets], dtype=np.int64)
        ok_s, ok_t = s >= 0, t >= 0
        if ok_s.any() and ok_t.any():
            d = dijkstra(self.graph, directed=True, indices=s[ok_s])
            out[np.ix_(ok_s, ok_t)] = d[:, t[ok_t]]
        return out

    def path(self, start, goal, start_radius=0.6, goal_radius=0.5, goal_dz=0.35):
        """(path (N, 3) of node centres, J_geo) from start to the reachable node nearest to goal
        within goal_radius (xy) and goal_dz (z); (None, inf) if there is none."""
        s = self.node(start, start_radius)
        if s < 0:
            return None, float('inf')
        dist, pred = self._search(s)
        pos = self.positions()
        d_goal = np.hypot(pos[:, 0] - goal[0], pos[:, 1] - goal[1])
        ok = np.isfinite(dist) & (d_goal <= goal_radius) & (np.abs(pos[:, 2] - goal[2]) <= goal_dz)
        if not ok.any():
            return None, float('inf')
        cand = np.flatnonzero(ok)
        t = int(cand[np.argmin(d_goal[cand] + 1e-3 * dist[cand])])
        nodes = _backtrack(pred, s, t)
        if nodes is None:
            return None, float('inf')
        return pos[nodes], float(dist[t])

    def near_stairs(self, near=0.3):
        """Per node: within `near` (xy) and one step (step_layers) in z of a stair voxel, so the
        landing right above the last step counts too."""
        if self._near_stairs is None:
            out = np.zeros(self.n, dtype=bool)
            if self.stairs is not None and self.stairs.any() and self.n:
                from scipy.spatial import cKDTree
                r = max(1, int(round(near / self.res)))
                sc = np.array([r, r, max(self.step_layers, 1)], dtype=np.float64)
                tree = cKDTree(np.argwhere(self.stairs) / sc)
                pos = np.argwhere(self.index >= 0)
                d, _ = tree.query(pos / sc, k=1, p=np.inf, distance_upper_bound=1.0 + 1e-9)
                out[self.index[tuple(pos.T)]] = np.isfinite(d)
            self._near_stairs = out
        return self._near_stairs

    def climb(self, start, center=None, radius=4.0, min_rise=0.1, min_stretch=0.3, landing=1.0,
              exclude=(), exclude_radius=0.3):
        """Next stretch of a stair climb on this graph (mode s): the path to the highest node near
        the stair voxels reachable from start, within radius (xy) of center (default: start) and
        at least min_rise above start. When that node is closer than min_stretch (the robot stands
        at the top edge, where z_r still reads the last step under half its footprint), the
        stretch goes on up to `landing` m onto its level (the upper floor). (None, None) when
        there is no such stretch: top reached / nothing known. Nodes within exclude_radius of a
        point in `exclude` (stretch ends the robot could not reach) are skipped. The trusted part
        of a staircase grows as z_r rises (VoxelMapper.trusted_stairs), so a robot that goes
        there, re-plans and goes on climbs the stairs stretch by stretch."""
        start = np.asarray(start, dtype=np.float64)
        s = self.node(start, 0.6)
        if s < 0:
            return None, None
        dist, pred = self._search(s)
        pos = self.positions()
        c = start if center is None else np.asarray(center, dtype=np.float64)
        ok = (np.isfinite(dist) & self.near_stairs()
              & (np.hypot(pos[:, 0] - c[0], pos[:, 1] - c[1]) <= radius))
        for e in np.asarray(exclude, dtype=np.float64).reshape(-1, 3):
            ok &= np.linalg.norm(pos - e, axis=1) > exclude_radius
        if not ok.any():
            return None, None
        cand = np.flatnonzero(ok)
        top = cand[pos[cand, 2] >= pos[cand, 2].max() - 1e-6]
        t = int(top[np.argmin(dist[top])])
        if pos[t, 2] - start[2] < min_rise:
            return None, None
        if np.hypot(*(pos[t, :2] - start[:2])) < min_stretch:
            level = np.isfinite(dist) & (np.abs(pos[:, 2] - pos[t, 2]) <= self.res + 1e-6)
            dxy = np.hypot(pos[:, 0] - start[0], pos[:, 1] - start[1])
            far = np.flatnonzero(level & (dxy <= landing))
            if not len(far) or dxy[far].max() < min_stretch:
                return None, None
            t = int(far[np.argmax(dxy[far])])
        nodes = _backtrack(pred, s, t)
        if nodes is None:
            return None, None
        return pos[nodes], float(dist[t])
