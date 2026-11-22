# Spatial value map (TravExplorer Sec. V-B, eq. 7).
#
# V^t = (s^t, r^t) over robot-reachable traversable voxels: s_k in [0, 1] the accumulated
# language-conditioned value of voxel v_k, r_k in [0, 1] its accumulated reliability. Each image
# gets one image-text relevance s^ (BLIP-2); it is projected only onto the traversable voxels
# visible in that image, each with the observation weight
#   w_k = cos^2(pi theta_k / theta_FOV)
# of its horizontal angle theta_k to the camera heading (1 in the image centre, 0 at the FOV
# boundary), and fused by confidence:
#   s_k <- (r_k s_k + w_k s^) / (r_k + w_k),   r_k <- (r_k^2 + w_k^2) / (r_k + w_k)          (7)
# A frontier's semantic utility is the maximum value in its local traversable neighbourhood.
#
# Stored sparsely (sorted voxel keys + values), so only seen voxels cost memory. Pure numpy.

import math

import numpy as np


def view_weight(theta, fov):
    """w = cos^2(pi theta / theta_FOV) for |theta| <= theta_FOV / 2, else 0."""
    theta = np.asarray(theta, dtype=np.float64)
    w = np.cos(math.pi * theta / fov) ** 2
    return np.where(np.abs(theta) <= 0.5 * fov, w, 0.0)


class ValueMap:

    def __init__(self):
        self.keys = np.zeros(0, dtype=np.int64)        # sorted voxel keys (flat index of the voxel map)
        self.s = np.zeros(0)
        self.r = np.zeros(0)

    def __len__(self):
        return len(self.keys)

    def update(self, keys, weights, score, min_weight=1e-3):
        """Fuse one image's score s^ into the voxels `keys` with observation weights (eq. 7).
        A voxel seen in several pixels gets the mean weight of its pixels."""
        keys = np.asarray(keys, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        ok = weights > min_weight
        if not ok.any():
            return 0
        u, inv = np.unique(keys[ok], return_inverse=True)
        w = np.bincount(inv, weights=weights[ok]) / np.bincount(inv)
        score = float(np.clip(score, 0.0, 1.0))
        pos = np.searchsorted(self.keys, u)
        old = (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == u) \
            if len(self.keys) else np.zeros(len(u), dtype=bool)
        p, wo = pos[old], w[old]
        r0, s0 = self.r[p], self.s[p]
        self.s[p] = (r0 * s0 + wo * score) / (r0 + wo)
        self.r[p] = (r0 * r0 + wo * wo) / (r0 + wo)
        if (~old).any():                                 # r = 0 before: s = s^, r = w
            nk, nw = u[~old], w[~old]
            keys_all = np.concatenate([self.keys, nk])
            order = np.argsort(keys_all, kind='stable')
            self.keys = keys_all[order]
            self.s = np.concatenate([self.s, np.full(len(nk), score)])[order]
            self.r = np.concatenate([self.r, nw])[order]
        return len(u)

    def get(self, keys):
        """(s, r) of keys; 0 for voxels never seen."""
        keys = np.asarray(keys, dtype=np.int64)
        s, r = np.zeros(len(keys)), np.zeros(len(keys))
        if len(self.keys):
            pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
            hit = self.keys[pos] == keys
            s[hit], r[hit] = self.s[pos[hit]], self.r[pos[hit]]
        return s, r


def utilities(points, values, queries, radius=1.0, dz=0.3):
    """Semantic utility of each query point (x, y, z): the maximum value of the value-map voxels
    (points, values) within `radius` (xy) and `dz` (z); 0 where none was seen."""
    out = np.zeros(len(queries))
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return out
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :2])
    values = np.asarray(values, dtype=np.float64)
    for i, q in enumerate(np.asarray(queries, dtype=np.float64).reshape(-1, 3)):
        idx = tree.query_ball_point(q[:2], radius)
        if idx:
            idx = np.asarray(idx)
            idx = idx[np.abs(points[idx, 2] - q[2]) <= dz]
            if len(idx):
                out[i] = values[idx].max()
    return out


def rviz_rainbow(v):
    """RGB in [0, 1] of RViz's rainbow colour map (PointCloud2 'Intensity', 'Use rainbow'):
    0 red -> yellow -> green -> cyan -> blue -> 1 magenta, as the VoxelValue display."""
    v = 1.0 - np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    h = v * 5.0 + 1.0
    i = np.floor(h)
    f = h - i
    f = np.where(i % 2 == 0, 1.0 - f, f)
    n = 1.0 - f
    one, zero = np.ones_like(n), np.zeros_like(n)
    r = np.select([i <= 1, i == 2, i == 3, i == 4], [n, zero, zero, n], one)
    g = np.select([i <= 1, i == 2, i == 3, i == 4], [zero, n, one, one], n)
    b = np.select([i <= 1, i == 2, i == 3, i == 4], [one, one, n, zero], zero)
    return np.stack([r, g, b], axis=-1)


def score_overlay(rgb, score, fx, cx, fov, alpha=0.45):
    """The RGB image (H x W x 3 uint8) tinted with the contribution of its image-text score to the
    value map: colour rviz_rainbow(score), opacity alpha * w(theta) per column, w the observation
    weight of eq. 7 (theta = horizontal angle of the column to the camera heading). Returns
    (image, w per column)."""
    w_px = rgb.shape[1]
    theta = np.arctan((np.arange(w_px) + 0.5 - cx) / fx)
    w = view_weight(theta, fov)
    a = (alpha * w)[None, :, None]
    col = 255.0 * rviz_rainbow(score)
    out = (1.0 - a) * rgb.astype(np.float64) + a * col
    return np.clip(out, 0, 255).astype(np.uint8), w
