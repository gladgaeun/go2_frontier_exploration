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
#
# Incremental update (following FUEL [16]): with B_m the AABB of the region where M_trav changed,
# clusters whose AABB intersects the expanded B_m are removed, and new clusters are grown with
# 26-connectivity from the candidates in the expanded B_m and in the removed clusters. Growing
# continues past the search box until every new cluster is complete, so the result is the same
# as extracting all frontiers from scratch.
#
# Pure numpy/scipy, no ROS imports.

import numpy as np
from scipy import ndimage

_CONN26 = np.ones((3, 3, 3), dtype=bool)


class FrontierCluster:
    __slots__ = ('id', 'voxels', 'centroid', 'aabb', 'visited')

    def __init__(self, cid, voxels, centroid, aabb):
        self.id = cid
        self.voxels = voxels          # C_i: flat voxel indices (sorted)
        self.centroid = centroid      # p_i: geometric centroid in the world frame
        self.aabb = aabb              # B_i: ((i0, j0, k0), (i1, j1, k1)), voxel index, [lo, hi)
        self.visited = False

    @property
    def size(self):
        return int(self.voxels.size)


class FrontierManager:

    def __init__(self, mapper, min_size=4, headroom_layers=None, grow_margin=8, boundary='source'):
        if boundary not in ('source', 'trav'):
            raise ValueError(f"boundary must be 'source' or 'trav', got {boundary!r}")
        self.boundary = boundary
        self.m = mapper
        self.min_size = int(min_size)
        self.k_h = int(headroom_layers) if headroom_layers else mapper.n_h
        self.grow_margin = int(grow_margin)
        self.clusters = {}            # id -> FrontierCluster
        self._next_id = 0

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
        src = self.m._source(sl)                                          # T
        trav = src & ~self.m.inflated[sl]                                   # M_trav
        bnd = trav if self.boundary == 'trav' else src
        nz = trav.shape[2]
        above = np.zeros_like(trav)                                         # M_trav within K_h above
        column = bnd.copy()                                                 # column within +-K_h
        for d in range(1, min(kh, nz - 1) + 1):
            above[:, :, :nz - d] |= trav[:, :, d:]
            column[:, :, :nz - d] |= bnd[:, :, d:]
            column[:, :, d:] |= bnd[:, :, :nz - d]
        # neighbour columns; outside the map is not traversable (cells cut off by the box
        # padding are inside the map only where the box was clipped at the map edge)
        col = np.pad(column, ((1, 1), (1, 1), (0, 0)), constant_values=False)
        open_nb = np.zeros_like(trav)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            open_nb |= ~col[1 + dx:col.shape[0] - 1 + dx, 1 + dy:col.shape[1] - 1 + dy, :]
        cand = trav & ~above & open_nb
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
        return [c.id for c in removed], added

    def extract_all(self):
        """All frontier clusters from scratch (no state change): list of (voxels, centroid, aabb)."""
        shape = self.m.logodds.shape
        labels, n = ndimage.label(self.candidates((0, 0, 0), shape), structure=_CONN26)
        return self._clusters_from(labels, n, (0, 0, 0))

    def reset(self):
        self.clusters.clear()
        return self.update(((0, 0, 0), self.m.logodds.shape))

    # -- outputs ------------------------------------------------------------------
    def voxels(self):
        if not self.clusters:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([c.voxels for c in self.clusters.values()])

    def centroids(self):
        return np.array([c.centroid for c in self.clusters.values()]).reshape(-1, 3)
