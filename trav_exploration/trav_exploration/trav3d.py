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


def graph_from_arrays(trav, obst, res, step_layers=5, reach=2, lambda_esdf=0.1, d_safe=0.5, d_obs=None,
                      far_reach=0, far_from=None):
    """support_graph on given arrays: trav = M_trav voxels (nodes), obst = obstacle voxels (d_obs),
    or d_obs itself (same shape as trav).

    Built from the node list (one gather per neighbour offset), not by shifting the whole box: the
    graph of the whole map (SupportGraph) spans every floor and the stairs between them, a box of
    millions of voxels with a few percent of them nodes.

    far_reach / far_from (bool, shape of trav): nodes in far_from are also linked within
    far_reach cells in xy (both ways): on the stairs a segmentation mask that misses the nosings
    leaves them occupied, their inflation covers most of the tread below, and only 1-2 cells of
    each tread stayed traversable, 0.15-0.2 m apart (00800 with SAM3, going down): with reach 2
    the staircase was not connected to the floor (the paper's foothold-scale neighbourhoods)."""
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
    if far_reach > reach and far_from is not None:
        far = far_from[trav]                           # per node
        src = np.flatnonzero(far)
        fflat = flat[src]
        far_off = [(di, dj) for di in range(-far_reach, far_reach + 1) for dj in range(-far_reach, far_reach + 1)
                   if reach * reach < di * di + dj * dj <= far_reach * far_reach]
        pad2 = np.array([far_reach, far_reach, dz])
        ip2 = np.full((sx + 2 * far_reach, sy + 2 * far_reach, sz + 2 * dz), -1, dtype=np.int64)
        ip2[far_reach:far_reach + sx, far_reach:far_reach + sy, dz:dz + sz] = index
        st2 = np.array([ip2.shape[1] * ip2.shape[2], ip2.shape[2], 1])
        f2 = (pos[src] + pad2) @ st2
        lin2 = ip2.reshape(-1)
        del fflat
        for di, dj in far_off:
            for dk in range(-dz, dz + 1):
                nb = lin2[f2 + (di * st2[0] + dj * st2[1] + dk)]
                ok = nb >= 0
                if not ok.any():
                    continue
                a, b = src[ok], nb[ok]
                c = res * math.sqrt(di * di + dj * dj + dk * dk)
                rows.append(a)                         # far node -> any node
                cols.append(b)
                vals.append(c + lambda_esdf * phi_n[b])
                back = ~far[b]                         # and back, unless b adds it itself
                rows.append(b[back])
                cols.append(a[back])
                vals.append(c + lambda_esdf * phi_n[a[back]])
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

    def __init__(self, snap, step_layers=5, reach=2, lambda_esdf=0.1, d_safe=0.5, stair_reach=4):
        self.res, self.origin, self.lo = snap['res'], snap['origin'], snap['lo']
        self.step_layers = step_layers
        trav, obst = snap['trav'], snap['obst']
        st = snap.get('stairs')
        self.graph, self.index, self.trav = graph_from_arrays(
            trav, obst, self.res, step_layers, reach, lambda_esdf, d_safe, d_obs=snap.get('d_obs'),
            far_reach=stair_reach, far_from=st & trav if st is not None else None)
        self.stairs = snap.get('stairs')
        self.n = int(self.trav.sum())
        # d_obs per node (distance to O_inf as Nav2 sees it + the O_inf radius, eq. 16): the
        # clearance of goals placed on the graph (target_goal, clear_node)
        d = snap.get('d_obs')
        self.d_obs = d[self.trav].astype(np.float64) if d is not None else np.full(self.n, np.inf)
        self._pos = None
        self._near_stairs = None
        self._dist = {}               # source node -> (dist, pred)

    @staticmethod
    def snapshot(m, visited=None, tight=None, margin=0.5, d_max=1.0, stair_margin=True):
        """Copy of the arrays the graph needs (call on the thread that writes the map).
        visited / tight: {layer: (nx, ny) bool} (VoxelMapper.visited_layers / tight_layers).
        d_obs is exact up to d_max (>= d_safe: beyond it Phi = 0, eq. 16).
        stair_margin=False: stair voxels lose only the O_inf of their own layer, not the one-cell
        margin around the O_inf of their layer and the ones right above / below (the descent's
        fallback: from above, SAM3 misses the nosings, their inflation leaves treads 1-2 cells
        wide, and the margin removed them; 00800, stuck 0.8 m above the lower floor)."""
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
        if not stair_margin:
            trav |= src & m.stairs[bxy] & ~inf
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

    def cluster_costs(self, source, clusters, shape, source_radius=0.6):
        """J_geo from source to each voxel set: min over its voxels that are nodes (flat indices
        into a map of `shape`), and that voxel's centre. [(J, (x, y, z) or None)]; J inf when no
        voxel of the set is reachable (a frontier behind a wall: its own voxels are not connected
        to the robot, whatever lies within 0.5 m of its centroid)."""
        out = [(np.inf, None)] * len(clusters)
        if self.n == 0:
            return out
        s = self.node(source, source_radius)
        if s < 0:
            return out
        dist = self._search(s)[0]
        pos = self.positions()
        lim = np.array(self.index.shape)
        res = []
        for vox in clusters:
            if not len(vox):
                res.append((np.inf, None))
                continue
            ijk = np.stack(np.unravel_index(np.asarray(vox), shape), axis=1) - np.asarray(self.lo)
            ok = ((ijk >= 0) & (ijk < lim)).all(axis=1)
            nodes = self.index[tuple(ijk[ok].T)] if ok.any() else np.zeros(0, dtype=np.int64)
            nodes = nodes[nodes >= 0]
            if not nodes.size:
                res.append((np.inf, None))
                continue
            d = dist[nodes]
            b = int(np.argmin(d))
            res.append((float(d[b]), tuple(pos[nodes[b]])) if np.isfinite(d[b]) else (np.inf, None))
        return res

    def clear_node(self, node, min_d_obs, radius=0.5, dz=0.15, source=None):
        """A node near `node` with d_obs >= min_d_obs, for a goal Nav2 can stop at: `node` itself if
        it is clear, else the nearest clear node within radius (xy) and dz (z) of it, reachable
        from source when one is given (its node's search). -1 if there is none. The 3-D version of
        the goal clearance on trav_grid (snap_to_clear_cell), on the nodes of the support graph."""
        if node < 0:
            return -1
        if self.d_obs[node] >= min_d_obs:
            return int(node)
        pos = self.positions()
        p = pos[node]
        ok = (np.hypot(pos[:, 0] - p[0], pos[:, 1] - p[1]) <= radius) & (np.abs(pos[:, 2] - p[2]) <= dz) \
            & (self.d_obs >= min_d_obs)
        if source is not None:
            s = self.node(source, 0.6)
            if s < 0:
                return -1
            ok &= np.isfinite(self._search(s)[0])
        if not ok.any():
            return -1
        cand = np.flatnonzero(ok)
        return int(cand[np.argmin(np.hypot(pos[cand, 0] - p[0], pos[cand, 1] - p[1]))])

    def target_goal(self, source, pts, r_min=0.35, r_goal=0.75, k_min=3, min_d_obs=0.0,
                    below=1.5, above=0.3, source_radius=0.6):
        """Goal node for a target object (O_tar) on the support graph: among the nodes from which
        at least k_min of its points lie within r_goal (xy), none closer than r_min (not in the
        object), at most `below` under and `above` over its lowest point (the floor it stands on,
        or under a TV on the wall), with d_obs >= min_d_obs, the one the robot reaches cheapest.
        Nodes are M_trav voxels at their own height: the floor around the object as the 3-D map
        holds it, not the slice of trav_grid at z_r (00813: the floor 15 cm under z_r was not on
        trav_grid, and no chair had a free cell near it).
        Returns dict(node, J, n_near, n_ring, n_clear, n_reach): node -1 when none is reachable,
        the counts tell at which condition the candidates ran out."""
        out = dict(node=-1, J=float('inf'), n_near=0, n_ring=0, n_clear=0, n_reach=0)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if self.n == 0 or not len(pts):
            return out
        from scipy.spatial import cKDTree
        pos = self.positions()
        z0 = pts[:, 2].min()
        band = (pos[:, 2] >= z0 - below) & (pos[:, 2] <= z0 + above)
        cand = np.flatnonzero(band)
        if not cand.size:
            return out
        tree = cKDTree(pts[:, :2])
        d, _ = tree.query(pos[cand, :2], k=1, distance_upper_bound=r_goal + 1e-6)
        near = cand[(d >= r_min) & (d <= r_goal)]
        out['n_near'] = int(near.size)
        if not near.size:
            return out
        n_in = np.array([len(x) for x in tree.query_ball_point(pos[near, :2], r_goal)])
        ring = near[n_in >= k_min]
        out['n_ring'] = int(ring.size)
        clear = ring[self.d_obs[ring] >= min_d_obs]
        out['n_clear'] = int(clear.size)
        if not clear.size:
            return out
        s = self.node(source, source_radius)
        if s < 0:
            return out
        dist = self._search(s)[0][clear]
        ok = np.isfinite(dist)
        out['n_reach'] = int(ok.sum())
        if ok.any():
            b = int(np.argmin(np.where(ok, dist, np.inf)))
            out['node'], out['J'] = int(clear[b]), float(dist[b])
        return out

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

    def reach_near(self, start, pts, radius=0.5, dz=0.3, highest=False, lowest=False, min_xy=0.0):
        """(node, J_geo) of the cheapest node reachable from start within radius (xy) and dz (z)
        of any of pts (e.g. the voxels of a frontier cluster); with highest, the highest such
        node (the cheapest of them): a frontier at the edge of the trusted stairs lies in O_inf,
        and the cheapest node near it was the floor at the foot of the stairs; lowest likewise
        going down. min_xy: only nodes that far (xy) from start (a stand 0.1 m ahead counts as
        reached by the path follower, and the robot stayed on its step). (-1, inf) if none."""
        s = self.node(start, 0.6, dz=0.15)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if s < 0 or not len(pts):
            return -1, float('inf')
        from scipy.spatial import cKDTree
        dist, _ = self._search(s)
        pos = self.positions()
        d, i = cKDTree(pts[:, :2]).query(pos[:, :2], k=1, distance_upper_bound=radius)
        ok = np.isfinite(d) & np.isfinite(dist)
        ok[ok] &= np.abs(pos[ok, 2] - pts[i[ok], 2]) <= dz
        if min_xy > 0.0:
            st = np.asarray(start, dtype=np.float64)
            ok &= np.hypot(pos[:, 0] - st[0], pos[:, 1] - st[1]) >= min_xy
        if not ok.any():
            return -1, float('inf')
        cand = np.flatnonzero(ok)
        if highest:
            cand = cand[pos[cand, 2] >= pos[cand, 2].max() - 1e-6]
        elif lowest:
            cand = cand[pos[cand, 2] <= pos[cand, 2].min() + 1e-6]
        t = int(cand[np.argmin(dist[cand])])
        return t, float(dist[t])

    def path_to(self, start, node):
        """Path (N, 3) from start to a node, or None."""
        s = self.node(start, 0.6, dz=0.15)
        if s < 0 or node < 0:
            return None
        _, pred = self._search(s)
        nodes = _backtrack(pred, s, int(node))
        return None if nodes is None else self.positions()[nodes]

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

    def flat_area(self, p, dz=0.1, seed_radius=0.6, near=0.3):
        """Area [m^2] of the flat floor the robot at p is on or next to: the nodes connected on the
        graph to the nodes within seed_radius (xy) of p, through nodes within dz of their height
        and not near the stairs (near_stairs). Counted as xy cells. A floor reached at the end of a
        climb has metres of it; a landing between two flights has little or none (its cells next
        to the steps are near the stairs)."""
        if self.n == 0:
            return 0.0
        from scipy.sparse.csgraph import connected_components
        pos = self.positions()
        p = np.asarray(p, dtype=np.float64)
        s = self.node(p, seed_radius, dz=0.15)
        if s < 0:
            return 0.0
        z0 = pos[s, 2]
        ok = (np.abs(pos[:, 2] - z0) <= dz + 1e-9) & ~self.near_stairs(near)
        seeds = np.flatnonzero(ok & (np.hypot(pos[:, 0] - p[0], pos[:, 1] - p[1]) <= seed_radius))
        if not len(seeds):
            return 0.0
        sel = np.flatnonzero(ok)
        sub = self.graph[sel][:, sel]
        _, lab = connected_components(sub, directed=False)
        where = np.full(self.n, -1, dtype=np.int64)
        where[sel] = np.arange(len(sel))
        keep = np.isin(lab, np.unique(lab[where[seeds]]))
        cells = np.unique(np.floor(pos[sel[keep], :2] / self.res).astype(np.int64), axis=0)
        return float(len(cells)) * self.res ** 2

    def step_off(self, start, down=False, min_xy=0.4, max_xy=2.0, max_rise=0.5, min_rise=0.1, exclude=(),
                 exclude_radius=0.3):
        """Off the stairs onto the floor at their end: the cheapest reachable node away from the
        stairs (not near_stairs), 0.1-max_rise m below start (down) or above it, within
        min_xy-max_xy (xy) of start. (path, J_geo) or (None, None). For a climb that ended on
        the last step: the path follower counts a goal 0.1 m ahead as reached, and the robot
        stayed on the bottom step (00800 going down: z_r -2.96 m, the floor at -3.16 m), whose
        z_r is not a new floor."""
        start = np.asarray(start, dtype=np.float64)
        s = self.node(start, 0.6)
        if s < 0:
            return None, None
        dist, pred = self._search(s)
        pos = self.positions()
        rise = (start[2] - pos[:, 2]) if down else (pos[:, 2] - start[2])
        dxy = np.hypot(pos[:, 0] - start[0], pos[:, 1] - start[1])
        ok = (np.isfinite(dist) & ~self.near_stairs() & (rise >= min_rise) & (rise <= max_rise)
              & (dxy >= min_xy) & (dxy <= max_xy))
        for e in np.asarray(exclude, dtype=np.float64).reshape(-1, 3):
            ok &= np.linalg.norm(pos - e, axis=1) > exclude_radius
        if not ok.any():
            return None, None
        cand = np.flatnonzero(ok)
        t = int(cand[np.argmin(dist[cand])])
        nodes = _backtrack(pred, s, t)
        return (None, None) if nodes is None else (pos[nodes], float(dist[t]))

    def climb(self, start, center=None, radius=4.0, min_rise=0.1, min_stretch=0.3, landing=1.0,
              exclude=(), exclude_radius=0.3, step_off=False, down=False):
        """Next stretch of a stair climb on this graph (mode s): the path to the highest node near
        the stair voxels reachable from start, within radius (xy) of center (default: start) and
        at least min_rise above start. When that node is closer than min_stretch (the robot stands
        at the top edge, where z_r still reads the last step under half its footprint), the
        stretch goes on up to `landing` m onto its level (the upper floor). (None, None) when
        there is no such stretch: top reached / nothing known. Nodes within exclude_radius of a
        point in `exclude` (stretch ends the robot could not reach) are skipped. The trusted part
        of a staircase grows as z_r rises (VoxelMapper.trusted_stairs), so a robot that goes
        there, re-plans and goes on climbs the stairs stretch by stretch. With down, the same
        going down: the lowest node, at least min_rise below start, stepping off at the bottom."""
        start = np.asarray(start, dtype=np.float64)
        s = self.node(start, 0.6)
        if s < 0:
            return None, None
        dist, pred = self._search(s)
        pos = self.positions()
        sg = -1.0 if down else 1.0
        hz, hs = sg * pos[:, 2], sg * start[2]             # "height" in the direction of travel
        c = start if center is None else np.asarray(center, dtype=np.float64)
        ok = (np.isfinite(dist) & self.near_stairs()
              & (np.hypot(pos[:, 0] - c[0], pos[:, 1] - c[1]) <= radius))
        for e in np.asarray(exclude, dtype=np.float64).reshape(-1, 3):
            ok &= np.linalg.norm(pos - e, axis=1) > exclude_radius
        if not ok.any():
            return None, None
        cand = np.flatnonzero(ok)
        top = cand[hz[cand] >= hz[cand].max() - 1e-6]
        t = int(top[np.argmin(dist[top])])
        if hz[t] - hs < min_rise:
            if not step_off:
                return None, None
            # on the top step with nothing higher near the stairs yet (the landing not in the
            # graph from below; 00810: the climb ended on the 2.68 m step of a 2.93 m floor):
            # step off onto the highest reachable level within `landing`, away from the stairs
            near = self.near_stairs()
            if not near[s]:
                return None, None                      # already off the stairs: done
            dxy = np.hypot(pos[:, 0] - start[0], pos[:, 1] - start[1])
            okf = np.isfinite(dist) & ~near & (hz >= hs - self.res) & (dxy >= min_stretch) & (dxy <= landing)
            for e in np.asarray(exclude, dtype=np.float64).reshape(-1, 3):
                okf &= np.linalg.norm(pos - e, axis=1) > exclude_radius     # 00820: the same
            off = np.flatnonzero(okf)                                       # unreachable stand 13 times
            if not len(off):
                return None, None
            hi = off[hz[off] >= hz[off].max() - 1e-6]
            t = int(hi[np.argmin(dist[hi])])
            nodes = _backtrack(pred, s, t)
            return (None, None) if nodes is None else (pos[nodes], float(dist[t]))
        if np.hypot(*(pos[t, :2] - start[:2])) < min_stretch:
            level = np.isfinite(dist) & (np.abs(pos[:, 2] - pos[t, 2]) <= self.res + 1e-6)
            for e in np.asarray(exclude, dtype=np.float64).reshape(-1, 3):
                level &= np.linalg.norm(pos - e, axis=1) > exclude_radius   # 00803: the same
            dxy = np.hypot(pos[:, 0] - start[0], pos[:, 1] - start[1])     # landing 7 times
            far = np.flatnonzero(level & (dxy <= landing))
            if not len(far) or dxy[far].max() < min_stretch:
                return None, None
            t = int(far[np.argmax(dxy[far])])
        nodes = _backtrack(pred, s, t)
        if nodes is None:
            return None, None
        return pos[nodes], float(dist[t])
