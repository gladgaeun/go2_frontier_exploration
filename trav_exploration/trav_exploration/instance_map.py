# Probabilistic semantic instance map (TravExplorer Sec. V-A, eq. 4-6).
#
# Single-frame open-vocabulary detections d_q = (P_q, s_q, q) (back-projected instance mask, its
# confidence, its query) are accumulated into persistent 3-D clusters y with a label q_y, a fused
# point set P_y (voxel-downsampled), an evidence count n_y and a fused confidence gamma_y:
#   associated (same label, centroid distance < assoc_dist):          positive evidence
#     P_y <- D(P_y U P_q),  n_y <- n_y + n_q,  gamma_y <- (n_y gamma_y + n_q s_q) / (n_y + n_q)   (4, 5)
#   not associated:                                                     a new cluster (n_q, s_q)
#   visible in the frame but not detected (only for the queries asked): negative evidence
#     n_y <- n_y + n~_y,  gamma_y <- n_y gamma_y / (n_y + n~_y)                                    (6)
# A cluster is used by the planner only when gamma_y >= min_conf (else it stays a tentative
# hypothesis); among the usable ones of a query, the belief gamma_y n_y ranks them.
#
# Score-weighted fusion (score_weight, on): no single-frame threshold decides what is evidence;
# every detection the segmenter returns (down to its noise floor, sam3_server --detect-confidence
# 0.05) is an observation, weighted by its own score: its points enter the fusion with weight
# w_q = n_q s_q, so a 0.2 detection moves gamma_y a quarter as much as a 0.8 one of the same size:
#     w_y <- w_y + n_q s_q,   gamma_y <- (w_y gamma_y + n_q s_q * s_q) / (w_y + n_q s_q)
#     negative evidence, eq. 6 as written: gamma_y <- n_y gamma_y / (n_y + n~_y), and w_y grows by
#     n~_y w_y / n_y (the cluster's mean weight per point). Weight 1 per undetected point instead
#     made it stronger than the score-weighted positives: a 0.8 chair missed every third frame
#     fell below min_conf.
# n_y stays the count of evidence points (eq. 4) for min_count and the belief. With score_weight
# off the weight is n_q itself, eq. 5 as written.
#
# Inconsistent observations are rejected (reject_ratio): a detection associated with a cluster
# but scored below reject_ratio x gamma_y (0.5: under half the cluster's confidence) is transient
# noise, not evidence: no fusion, no count, no frame; the cluster counts as seen in this frame (no
# negative evidence from it either). With the 0.05 noise floor the weak masks SAM3 lays over a
# real object in most frames held a sofa at gamma 0.50-0.51 (00803), on and off the target set.
#
# Choices not given in the paper:
#   * n_q = |D(P_q)|, the points of the detection after the voxel downsampling (voxel), so that
#     the negative evidence n~_y (visible voxels of P_y) is on the same scale.
#   * Association also when the point sets overlap (overlap of the smaller one): a staircase or a
#     sofa seen from different places has centroids metres apart, and centroid distance alone
#     split it into several clusters.
#   * A cluster needs min_count evidence points as well, so a single small detection is not used.
#   * ... and detections in min_detections frames: one SAM3 false positive (confidence 0.55, 35
#     points) became a reliable "refrigerator" on the upper floor of 00800 as the robot went down
#     the stairs, was never looked at again (no negative evidence), and sent the robot back up
#     from the ground floor (eq. 9: targets first).
#
# Pure numpy, no ROS.

import itertools

import numpy as np

_OFF = 1 << 20                 # key packing: 21 bits per axis (+-1e6 voxels)


class Instance:
    __slots__ = ('id', 'label', 'keys', 'n', 'w', 'gamma', 'n_pos', 'n_neg', 'last_seen')

    def __init__(self, id_, label, keys, n, gamma, t, w=None):
        self.id, self.label, self.keys = id_, label, keys
        self.n, self.gamma = float(n), float(gamma)
        self.w = float(n if w is None else w)    # fusion weight of the evidence so far
        self.n_pos, self.n_neg = 1, 0            # detections / frames of negative evidence (log)
        self.last_seen = t

    @property
    def belief(self):
        return self.gamma * self.n


class InstanceMap:

    def __init__(self, voxel=0.1, assoc_dist=1.0, overlap=0.2, min_conf=0.5, min_count=30, min_visible=5,
                 min_detections=2, score_weight=True, reject_ratio=0.5):
        self.voxel = float(voxel)
        self.assoc_dist = float(assoc_dist)
        self.overlap = float(overlap)
        self.min_conf = float(min_conf)
        self.min_count = float(min_count)
        self.min_visible = int(min_visible)
        self.min_detections = int(min_detections)
        self.score_weight = bool(score_weight)
        self.reject_ratio = float(reject_ratio)
        self.n_rejected = 0
        self.instances = []
        self._ids = itertools.count()

    # -- voxel keys -------------------------------------------------------------------------
    def keys(self, pts):
        """Sorted unique voxel keys of points (N, 3): the downsampling D(.)."""
        k = np.floor(np.asarray(pts, dtype=np.float64).reshape(-1, 3) / self.voxel).astype(np.int64) + _OFF
        return np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2])

    def points(self, keys):
        """Voxel centres of keys (N, 3)."""
        k = np.stack([(keys >> 42) & 0x1FFFFF, (keys >> 21) & 0x1FFFFF, keys & 0x1FFFFF], axis=1) - _OFF
        return (k + 0.5) * self.voxel

    def centroid(self, inst):
        return self.points(inst.keys).mean(axis=0)

    # -- update -----------------------------------------------------------------------------
    def update(self, detections, queries=(), visible=None, t=0.0):
        """One frame. detections: [(label, points (N, 3) in the map frame, score)]; queries: the
        labels asked in this frame (negative evidence only for those); visible(points) -> number
        of those points in view and not occluded in this frame. Returns the ids of the clusters
        that got a detection."""
        matched = set()
        for label, pts, score in detections:
            k = self.keys(pts)
            if not len(k):
                continue
            c = self.points(k).mean(axis=0)
            best, best_d = None, np.inf
            for inst in self.instances:
                if inst.label != label:
                    continue
                d = float(np.linalg.norm(self.centroid(inst) - c))
                ov = len(np.intersect1d(inst.keys, k, assume_unique=True)) / min(len(inst.keys), len(k))
                if (d < self.assoc_dist or ov >= self.overlap) and d < best_d:
                    best, best_d = inst, d
            n_q = len(k)
            score = float(score)
            if best is not None and score < self.reject_ratio * best.gamma:
                self.n_rejected += 1             # inconsistent with the cluster: transient noise
                matched.add(best.id)
                continue
            w_q = n_q * score if self.score_weight else float(n_q)   # weight of this observation
            if best is None:
                best = Instance(next(self._ids), label, k, n_q, score, t, w=w_q)
                self.instances.append(best)
            else:
                best.n += n_q                                                      # (4)
                if w_q > 0.0:
                    best.gamma = (best.w * best.gamma + w_q * score) / (best.w + w_q)   # (5)
                    best.w += w_q
                best.keys = np.union1d(best.keys, k)
                if best.id not in matched:                                       # frames, not masks
                    best.n_pos += 1
                best.last_seen = t
            matched.add(best.id)
        if visible is not None:
            for inst in self.instances:
                if inst.label not in queries or inst.id in matched:
                    continue
                v = int(visible(self.points(inst.keys)))
                if v < self.min_visible:
                    continue
                # (6) as written: gamma_y <- n_y gamma_y / (n_y + n~_y); the weight grows by the
                # cluster's mean weight per point, so negative evidence keeps the paper's strength
                inst.gamma = inst.n * inst.gamma / (inst.n + v)
                inst.w += v * inst.w / inst.n if inst.n > 0 else 0.0
                inst.n += v
                inst.n_neg += 1
        return matched

    # -- queries ----------------------------------------------------------------------------
    def reliable(self, label=None):
        """Clusters the planner may use (gamma >= min_conf, n >= min_count, n_pos >= min_detections),
        by belief."""
        out = [i for i in self.instances if (label is None or i.label == label)
               and i.gamma >= self.min_conf and i.n >= self.min_count and i.n_pos >= self.min_detections]
        return sorted(out, key=lambda i: -i.belief)

    def stair_landmark(self, inst, part=0.25):
        """(entrance x, y, z, top x, y, z, points) of a stair cluster: the entrance is the mean
        xy of its lowest `part` m (the foot of the stairs, z its lowest point), the top the same
        for its highest part (as VoxelMapper.stair_landmarks)."""
        c = self.points(inst.keys)
        low = c[c[:, 2] <= c[:, 2].min() + part]
        high = c[c[:, 2] >= c[:, 2].max() - part]
        return (*low[:, :2].mean(axis=0), c[:, 2].min(), *high[:, :2].mean(axis=0), c[:, 2].max(), len(c))
