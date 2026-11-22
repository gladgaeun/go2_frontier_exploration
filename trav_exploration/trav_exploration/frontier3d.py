# Traversable frontiers on the 3D voxel map (TravExplorer, Sec. III-C).
#
# Frontiers are extracted from M_trav. A voxel v in M_trav is a frontier candidate if
#   1. headroom: no M_trav voxel within K_h layers above v, and
#   2. boundary: a horizontal 4-neighbour of v has a vertical column that is not traversable.
# Candidates are grouped by 26-neighbour connectivity into clusters F_i = <C_i, p_i, B_i>
# (voxels, centroid, AABB); clusters smaller than min_size are discarded.
#
# Not given in the paper, chosen here:
#   - K_h defaults to the robot height in layers (h_max / resolution).
#   - "the column is not traversable" looks at the neighbour column within +-K_h layers of v, so
#     another floor far above or below does not hide a frontier (on one floor this is the same
#     as the whole column).
#   - It tests the traversability source T (M_g in geometric mode), not M_trav. Read against
#     M_trav, every support voxel next to an inflated obstacle would be a frontier (the inflated
#     ring around walls is not in M_trav), so a closed room would never run out of frontiers.
#     The paper describes the boundary as the one to "unexplored or temporarily non-traversable"
#     regions; the inflated ring around a wall is explored (it is in M_g). boundary='trav'
#     gives the literal reading.
#   - Outside the map counts as not traversable.
#   - need_unknown (default; the name is historical): the neighbour column must not be a known
#     obstacle: no occupied voxel in the collision band above v (dz_lo up to one step more), and v's layer
#     there not in O_inf. Without it, a support voxel next to the foot of a wall, a chair leg or a
#     sofa side (covered floor, never T), or next to the hidden strip of floor at the foot of a
#     wall (unknown but inside O_inf) was a frontier; the robot went there, nothing more could be
#     seen, and it was tried again until it went dormant (00831: five goals within 1.5 m).
#     Unknown and seen-free columns stay open (requiring an unknown voxel dropped the far
#     frontiers, whose floor grazing rays had marked free). Below v (the floor slab) does not
#     count. The paper's "temporarily non-traversable" regions are the stairs, handled below.
#   - Stair voxels (traversable only through M_s, VoxelMapper.semantic_only_at) are candidates
#     even inside O_inf: the steps above the trusted window (VoxelMapper.trusted_stairs) are not
#     trusted yet, they are obstacles and their inflation covers the last trusted steps, so the
#     staircase never had a frontier (semantic frontiers F_s, Sec. VI-A, which guide the stair
#     climb). Their boundary is a neighbour column with no T and no wall within +-0.25 m (one
#     step; within K_h the floor below the steps counted) but with an unknown voxel or a stair
#     voxel not trusted yet with 0.4 m free above it (a railing is labelled stairs too): the sides of the staircase (walls, or the
#     void seen next to the railing) are not frontiers.
#
# Incremental update (following FUEL [16]): with B_m the AABB of the region where M_trav changed,
# clusters whose AABB intersects the expanded B_m are removed, and new clusters are grown with
# 26-connectivity from the candidates in the expanded B_m and in the removed clusters. Growing
# continues past the search box until every new cluster is complete, so the result is the same
# as extracting all frontiers from scratch.
#
# Visited (dormant) flags: the paper resets them with the clusters an update removes. Here a new
# cluster keeps the flag when it is the same frontier re-segmented (_same): at least keep_share of
# the old voxels lie within one voxel of it, and at least keep_share of it within one voxel of the
# old ones. Seen again from elsewhere, the edge of a frontier under a table shifts by a voxel and
# grows a few; with exact voxels required (80 %) the flag went at once and the robot drove back
# to the same two frontiers again and again (00803). When the space there changed (unknown seen
# free or occupied), the old frontier voxels are gone and the flag goes: the frontier is a
# candidate again; merged into a new open frontier, most of the new cluster is new and it is not
# dormant either. mark_visited(id) follows the clusters an update replaced (successor: the new
# cluster with most of the old one's voxels), so an id sent by
# frontier_goal_node a moment ago still finds its cluster. Failed goals (mark_failed: Nav2 gave up,
# the guard stopped the robot, no Nav2 path) are counted per cluster the same way, and at
# max_attempts the cluster goes dormant (the paper's repeated failures -> F_dor); the count goes
# with the flag when the voxels change.
#
# Pure numpy/scipy, no ROS imports.

import numpy as np
from scipy import ndimage

_CONN26 = np.ones((3, 3, 3), dtype=bool)


class FrontierCluster:
    __slots__ = ('id', 'voxels', 'centroid', 'aabb', 'visited', 'fails', 'anchor')

    def __init__(self, cid, voxels, centroid, aabb):
        self.id = cid
        self.voxels = voxels          # C_i: flat voxel indices (sorted)
        self.centroid = centroid      # p_i: geometric centroid in the world frame
        self.aabb = aabb              # B_i: ((i0, j0, k0), (i1, j1, k1)), voxel index, [lo, hi)
        self.visited = False
        self.fails = 0                # goals toward it that were not reached (mark_failed)
        self.anchor = None            # its voxels when it went dormant: the flag stays while
                                      # the frontier is still those voxels (not drifting away)

    @property
    def size(self):
        return int(self.voxels.size)


class FrontierManager:

    def __init__(self, mapper, min_size=4, headroom_layers=None, grow_margin=8, boundary='source',
                 keep_share=0.6, need_unknown=True):
        if boundary not in ('source', 'trav'):
            raise ValueError(f"boundary must be 'source' or 'trav', got {boundary!r}")
        self.boundary = boundary
        self.m = mapper
        self.min_size = int(min_size)
        self.k_h = int(headroom_layers) if headroom_layers else mapper.n_h
        self.k_stair = max(1, int(round(0.25 / mapper.res)))     # stair boundary: +-one step (0.25 m)
        self.k_step_head = max(1, int(round(0.4 / mapper.res)))  # free above a step to come
        self.grow_margin = int(grow_margin)
        self.clusters = {}            # id -> FrontierCluster
        self._next_id = 0
        self.keep_share = float(keep_share)
        self.need_unknown = bool(need_unknown)
        self.successor = {}           # removed id -> id of the new cluster with most of its voxels

    # -- candidates ---------------------------------------------------------------
    def _box(self, lo, hi, pad):
        shape = self.m.logodds.shape
        return (tuple(max(l - p, 0) for l, p in zip(lo, pad)),
                tuple(min(h + p, n) for h, p, n in zip(hi, pad, shape)))

    def candidates(self, lo, hi):
        """Frontier candidate mask of the box [lo, hi) (voxel indices)."""
        kh = self.k_h
        elo, ehi = self._box(lo, hi, (1, 1, kh))
        sl = tuple(slice(a, b) for a, b in zip(elo, ehi))
        m = self.m
        src = m._source(sl)                                                 # T
        trav = src & ~m.inflated[sl]                                        # M_trav
        stair = src & m.stairs[sl] & ~m.trav_g[sl]                          # only through M_s
        base = trav | stair
        bnd = trav if self.boundary == 'trav' else src
        nz = trav.shape[2]

        def within(a, h=kh):                                                # any within +-h layers
            out = a.copy()
            for d in range(1, min(h, nz - 1) + 1):
                out[:, :, :nz - d] |= a[:, :, d:]
                out[:, :, d:] |= a[:, :, :nz - d]
            return out

        above = np.zeros_like(trav)                                         # base within K_h above
        for d in range(1, min(kh, nz - 1) + 1):
            above[:, :, :nz - d] |= base[:, :, d:]
        column = within(bnd)                                                # column within +-K_h

        def neighbours(a, fill):
            p = np.pad(a, ((1, 1), (1, 1), (0, 0)), constant_values=fill)
            return [p[1 + dx:p.shape[0] - 1 + dx, 1 + dy:p.shape[1] - 1 + dy, :]
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]

        # neighbour columns; outside the map is not traversable (cells cut off by the box
        # padding are inside the map only where the box was clipped at the map edge)
        open_nb = np.zeros_like(trav)
        if self.need_unknown:
            # the neighbour column must not be a known obstacle: no occupied voxel at v's layer up
            # to one step above, and v's layer there not in O_inf. Unknown or seen free (a ray
            # grazing the floor far away marks it free without a hit) are open: those are the
            # frontiers. Requiring an unknown voxel instead dropped those far ones (00821: 3
            # clusters instead of 12 after the look-around, exploration over at 64 s).
            # occupied in the collision band above v (dz_lo .. dz_lo + one step), as the
            # inflation reads it: a door threshold or a rug edge lower than dz_lo is no obstacle
            occ_ = (m.logodds[sl] >= m.l_occ) | m.blocked[sl]
            blocked = m.inflated[sl].copy()
            for d in range(m.dz_lo, min(m.dz_lo + self.k_stair, nz - 1) + 1):
                blocked[:, :, :nz - d] |= occ_[:, :, d:]
            for c, b in zip(neighbours(column, False), neighbours(blocked, False)):
                open_nb |= ~c & ~b
        else:
            for c in neighbours(column, False):
                open_nb |= ~c
        cand = trav & ~stair & ~above & open_nb
        if stair.any():
            lo_ = m.logodds[sl]
            occ = (lo_ >= m.l_occ) | m.blocked[sl]
            pending = m.stair_seen[sl] & ~src                               # stairs not trusted yet
            # a step to come has free space above it; a railing (labelled stairs too) does not:
            # with it, the side of the staircase was a row of frontiers
            occ_above = np.zeros_like(occ)
            for d in range(1, min(self.k_step_head, nz - 1) + 1):
                occ_above[:, :, :nz - d] |= occ[:, :, d:]
            step_next = pending & ~occ_above
            hs = self.k_stair            # one step: within K_h the floor below the steps counted
            wall = within(occ & ~src & ~pending, hs)
            unknown = within((lo_ == 0.0) | step_next, hs)
            col_s = within(bnd, hs)
            open_s = np.zeros_like(trav)
            for c, wl, un in zip(neighbours(col_s, False), neighbours(wall, True), neighbours(unknown, False)):
                open_s |= ~c & ~wl & un
            cand |= stair & ~above & open_s
        inner = tuple(slice(a - e, b - e) for a, b, e in zip(lo, hi, elo))
        return cand[inner]

    # -- extraction ---------------------------------------------------------------
    def _grow(self, lo, hi):
        """Label candidates in [lo, hi); enlarge the box until no component touches its
        (inner) boundary. Returns (labels, n, lo, hi)."""
        shape = self.m.logodds.shape
        while True:
            labels, n = ndimage.label(self.candidates(lo, hi), structure=_CONN26)
            grow_lo, grow_hi = list(lo), list(hi)
            for ax in range(3):
                idx = [slice(None)] * 3
                if lo[ax] > 0:
                    idx[ax] = 0
                    if labels[tuple(idx)].any():
                        grow_lo[ax] = max(lo[ax] - self.grow_margin, 0)
                if hi[ax] < shape[ax]:
                    idx[ax] = -1
                    if labels[tuple(idx)].any():
                        grow_hi[ax] = min(hi[ax] + self.grow_margin, shape[ax])
            if (tuple(grow_lo), tuple(grow_hi)) == (tuple(lo), tuple(hi)):
                return labels, n, lo, hi
            lo, hi = tuple(grow_lo), tuple(grow_hi)

    def _clusters_from(self, labels, n, lo):
        out = []
        if n == 0:
            return out
        shape = self.m.logodds.shape
        res = self.m.res
        objs = ndimage.find_objects(labels)
        for lab, sl in enumerate(objs, start=1):
            sub = labels[sl] == lab
            size = int(sub.sum())
            if size < self.min_size:
                continue
            ii, jj, kk = np.nonzero(sub)
            ii = ii + sl[0].start + lo[0]
            jj = jj + sl[1].start + lo[1]
            kk = kk + sl[2].start + lo[2]
            flat = np.sort(np.ravel_multi_index((ii, jj, kk), shape))
            centroid = np.array([self.m.origin_xy + (ii.mean() + 0.5) * res,
                                 self.m.origin_xy + (jj.mean() + 0.5) * res,
                                 self.m.origin_z + (kk.mean() + 0.5) * res])
            aabb = ((int(ii.min()), int(jj.min()), int(kk.min())),
                    (int(ii.max()) + 1, int(jj.max()) + 1, int(kk.max()) + 1))
            out.append((flat, centroid, aabb))
        return out

    def _add(self, flat, centroid, aabb):
        c = FrontierCluster(self._next_id, flat, centroid, aabb)
        self.clusters[c.id] = c
        self._next_id += 1
        return c

    @staticmethod
    def _intersects(a, b):
        return all(a[0][i] < b[1][i] and b[0][i] < a[1][i] for i in range(3))

    def update(self, box):
        """Update the frontiers after M_trav changed inside box = B_m (or None).

        Returns (removed ids, added clusters)."""
        if box is None:
            return [], []
        # candidacy depends on +-1 voxel in xy and +-K_h layers: expanded B_m
        elo, ehi = self._box(box[0], box[1], (1, 1, self.k_h))
        expanded = (elo, ehi)
        removed = [c for c in self.clusters.values() if self._intersects(c.aabb, expanded)]
        for c in removed:
            del self.clusters[c.id]                         # visited flags go with them
            elo = tuple(map(min, elo, c.aabb[0]))
            ehi = tuple(map(max, ehi, c.aabb[1]))
        labels, n, lo, hi = self._grow(elo, ehi)
        new = self._clusters_from(labels, n, lo)
        # a grown cluster can reach into a kept cluster: it replaces it
        if new:
            box_all = (lo, hi)
            touched = [c for c in self.clusters.values() if self._intersects(c.aabb, box_all)]
            new_vox = np.concatenate([f for f, _, _ in new])
            for c in touched:
                if np.isin(c.voxels, new_vox, assume_unique=True).any():
                    removed.append(c)
                    del self.clusters[c.id]
        added = [self._add(*x) for x in new]
        self._link(removed, added)
        return [c.id for c in removed], added

    def _near(self, a, b):
        """How many voxels of a lie within one voxel (26-neighbourhood) of a voxel of b."""
        nx, ny, nz = self.m.logodds.shape
        off = np.array([dx * ny * nz + dy * nz + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)], dtype=np.int64)
        grown = np.unique((b[:, None] + off[None, :]).reshape(-1))
        return int(np.isin(a, grown).sum())

    def _same_vox(self, a, b):
        """Voxel sets a (old) and b (new): the same frontier within one voxel (see _same)."""
        return (self._near(a, b) >= self.keep_share * a.size and
                self._near(b, a) >= self.keep_share * b.size)

    def _same(self, old, c):
        """Old cluster `old` and new cluster c are the same frontier, re-segmented: most of the old
        voxels are still there and c is mostly made of them (within one voxel: the edge of a
        frontier seen again from elsewhere shifts by a voxel and grows a few). When the space there
        changed (unknown seen), the old frontier voxels are gone; when c merged into a new open
        frontier, most of c is new."""
        if not self._intersects(self._pad(old.aabb), self._pad(c.aabb)):
            return False
        return (self._near(old.voxels, c.voxels) >= self.keep_share * old.size and
                self._near(c.voxels, old.voxels) >= self.keep_share * c.size)

    @staticmethod
    def _pad(aabb):
        (a, b) = aabb
        return (tuple(x - 1 for x in a), tuple(x + 1 for x in b))

    def _link(self, removed, added):
        """Successors of the removed clusters and the visited flags / failure counts the added
        ones keep (see _same)."""
        if not removed or not added:
            return
        for old in removed:
            best, best_n = -1, 0
            for i, c in enumerate(added):
                if not self._intersects(old.aabb, c.aabb):
                    continue
                n = int(np.isin(c.voxels, old.voxels, assume_unique=True).sum())
                if n > best_n:
                    best, best_n = i, n
            if best >= 0 and 2 * best_n >= old.size:
                self.successor[old.id] = added[best].id
            if old.visited or old.fails:
                for c in added:
                    if not self._intersects(self._pad(old.aabb), self._pad(c.aabb)):
                        continue
                    # dormant: compared with the voxels it had when it went dormant, so that
                    # small changes frame by frame do not add up to a different frontier
                    if old.visited and old.anchor is not None and self._same_vox(old.anchor, c.voxels):
                        c.visited, c.anchor = True, old.anchor
                    if old.fails and self._same(old, c):
                        c.fails = max(c.fails, old.fails)

    def resolve(self, cid):
        """The current cluster that cluster cid became (None when it is gone)."""
        seen = []
        while cid not in self.clusters and cid in self.successor and len(seen) < 10000:
            seen.append(cid)
            cid = self.successor[cid]
        for s in seen:                                     # path compression
            self.successor[s] = cid
        return self.clusters.get(cid)

    def mark_failed(self, cid, max_attempts=2):
        """A goal toward cluster cid (or the cluster it became) was not reached: count it; at
        max_attempts the cluster is visited (dormant) until its voxels change."""
        c = self.resolve(cid)
        if c is not None:
            c.fails += 1
            if c.fails >= max_attempts and not c.visited:
                c.visited, c.anchor = True, c.voxels.copy()
        return c

    def mark_visited(self, cid):
        """Set the visited (dormant) flag of cluster cid or of the cluster it became."""
        c = self.resolve(cid)
        if c is not None and not c.visited:
            c.visited, c.anchor = True, c.voxels.copy()
        return c

    def extract_all(self):
        """All frontier clusters from scratch (no state change): list of (voxels, centroid, aabb)."""
        shape = self.m.logodds.shape
        labels, n = ndimage.label(self.candidates((0, 0, 0), shape), structure=_CONN26)
        return self._clusters_from(labels, n, (0, 0, 0))

    def release_dormant(self):
        """Clear every visited (dormant) flag: the last resort before exploration ends with only
        dormant frontiers left (frontier_goal_node, dormant_release_max). Failure counts stay, so a
        cluster that fails again goes dormant at once."""
        n = 0
        for c in self.clusters.values():
            if c.visited:
                c.visited, c.anchor = False, None
                n += 1
        return n

    def reset(self):
        # every cluster intersects the whole map: all are removed and grown again (flags kept
        # where the voxels are the same)
        return self.update(((0, 0, 0), self.m.logodds.shape))

    # -- outputs ------------------------------------------------------------------
    def voxels(self):
        if not self.clusters:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([c.voxels for c in self.clusters.values()])

    def centroids(self):
        return np.array([c.centroid for c in self.clusters.values()]).reshape(-1, 3)
