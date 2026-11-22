# Unit tests for the probabilistic instance map (instance_map.py, TravExplorer Sec. V-A, eq. 4-6).
#   python3 -m pytest trav_exploration/test/test_instance_map.py

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.instance_map import InstanceMap  # noqa: E402


def box(lo, hi, step=0.05):
    xs, ys, zs = (a + step * (np.arange(int(round((b - a) / step))) + 0.5) for a, b in zip(lo, hi))
    return np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)


def test_keys_round_trip_and_downsampling():
    m = InstanceMap(voxel=0.1)
    pts = box((-1.0, 2.0, 0.0), (-0.6, 2.2, 0.1))         # 0.4 x 0.2 x 0.1 m -> 4 x 2 x 1 voxels
    k = m.keys(pts)
    assert len(k) == 8
    c = m.points(k)
    assert np.allclose(np.sort(c[:, 0]), np.repeat([-0.95, -0.85, -0.75, -0.65], 2))
    assert np.all(np.abs(m.keys(c) - k) == 0)


def test_new_cluster_then_weighted_fusion_eq_4_5():
    m = InstanceMap(voxel=0.1, min_count=1, score_weight=False)            # eq. 5 as written
    a = box((0, 0, 0), (0.5, 0.5, 0.5))                   # 125 voxels
    m.update([('chair', a, 0.8)], queries=('chair',))
    (y,) = m.instances
    assert y.label == 'chair' and y.n == 125 and abs(y.gamma - 0.8) < 1e-9
    b = box((0.2, 0, 0), (0.7, 0.5, 0.5))                 # shifted 0.2 m: associated
    m.update([('chair', b, 0.4)], queries=('chair',))
    assert len(m.instances) == 1
    assert y.n == 250 and abs(y.gamma - (125 * 0.8 + 125 * 0.4) / 250) < 1e-9        # (5)
    assert len(y.keys) == 7 * 5 * 5                                                   # D(P_y U P_q)


def test_other_label_or_far_away_starts_a_new_cluster():
    m = InstanceMap(voxel=0.1, assoc_dist=1.0)
    a = box((0, 0, 0), (0.3, 0.3, 0.3))
    m.update([('chair', a, 0.9)], queries=('chair',))
    m.update([('couch', a, 0.9)], queries=('chair', 'couch'))           # same place, other label
    m.update([('chair', a + [3.0, 0, 0], 0.9)], queries=('chair',))     # same label, 3 m away
    assert sorted(i.label for i in m.instances) == ['chair', 'chair', 'couch']


def test_overlapping_views_of_a_long_object_are_one_cluster():
    # a staircase seen part by part: centroids 1.5 m apart, the point sets overlap
    m = InstanceMap(voxel=0.1, assoc_dist=1.0, overlap=0.2)
    m.update([('stairs', box((0, 0, 0), (2.0, 1.0, 0.2)), 0.9)], queries=('stairs',))
    m.update([('stairs', box((1.0, 0, 0), (4.0, 1.0, 0.2)), 0.9)], queries=('stairs',))
    assert len(m.instances) == 1


def test_negative_evidence_eq_6_lowers_a_false_positive():
    m = InstanceMap(voxel=0.1, min_conf=0.5, min_count=10, min_detections=1, score_weight=False)
    a = box((0, 0, 0), (0.5, 0.5, 0.2))                   # 50 voxels
    m.update([('chair', a, 0.9)], queries=('chair',))
    (y,) = m.instances
    assert m.reliable('chair') == [y]
    m.update([], queries=('chair',), visible=lambda pts: len(pts))      # in view, not detected
    assert y.n == 100 and abs(y.gamma - 0.45) < 1e-9                    # (6): 50 * 0.9 / 100
    assert m.reliable('chair') == []                                    # tentative now
    # not asked in this frame / not in view: no evidence either way
    m.update([], queries=('couch',), visible=lambda pts: len(pts))
    m.update([], queries=('chair',), visible=lambda pts: 0)
    assert y.n == 100


def test_stable_detections_survive_occasional_misses():
    m = InstanceMap(voxel=0.1, min_conf=0.5, min_count=10)
    a = box((0, 0, 0), (0.5, 0.5, 0.2))
    for k in range(10):
        dets = [('chair', a, 0.8)] if k % 3 else []                     # missed every third frame
        m.update(dets, queries=('chair',), visible=lambda pts: len(pts))
    (y,) = m.instances
    assert y.gamma > 0.5 and m.reliable('chair') == [y]


def test_reliable_ranks_by_belief_and_needs_min_count():
    m = InstanceMap(voxel=0.1, min_conf=0.5, min_count=20, min_detections=1)
    small = box((0, 0, 0), (0.2, 0.2, 0.2))               # 8 voxels: below min_count
    big = box((5, 0, 0), (5.5, 0.5, 0.2))                 # 50 voxels
    mid = box((-5, 0, 0), (-4.7, 0.5, 0.2))               # 30 voxels, more confident
    m.update([('chair', small, 0.99), ('chair', big, 0.6), ('chair', mid, 0.95)], queries=('chair',))
    rel = m.reliable('chair')
    assert [len(i.keys) for i in rel] == [50, 30]          # belief 0.6 * 50 = 30 > 0.95 * 30 = 28.5


def test_stair_landmark_entrance_and_top():
    m = InstanceMap(voxel=0.1)
    steps = np.concatenate([box((0.3 * k, 0, 0.2 * k), (0.3 * k + 0.3, 1.0, 0.2 * k + 0.1)) for k in range(10)])
    m.update([('stairs', steps, 0.9)], queries=('stairs',))
    ex, ey, ez, tx, ty, tz, n = m.stair_landmark(m.instances[0])
    assert ex < 0.6 and abs(ey - 0.5) < 0.1 and ez < 0.1          # foot
    assert tx > 2.4 and tz > 1.7 and n == len(m.instances[0].keys)


def test_one_frame_is_not_reliable_two_frames_are():
    # a single false positive must not become a target (00800: a "refrigerator" seen in one frame
    # on the upper floor sent the robot back up the stairs)
    m = InstanceMap(voxel=0.1, min_conf=0.5, min_count=10)
    a = box((0, 0, 0), (0.5, 0.5, 0.2))
    m.update([('chair', a, 0.9), ('chair', a, 0.9)], queries=('chair',))   # two masks, one frame
    (y,) = m.instances
    assert y.n_pos == 1 and m.reliable('chair') == []
    m.update([('chair', a, 0.9)], queries=('chair',), t=1.0)
    assert y.n_pos == 2 and m.reliable('chair') == [y]


def test_score_weighted_fusion_low_scores_move_gamma_little():
    """Every detection is an observation, weighted by its score: a weak one (0.2) pulls a
    confident cluster down far less than with eq. 5's point weights, a strong one lifts a weak
    cluster quickly."""
    a = box((0, 0, 0), (0.5, 0.5, 0.5))                   # 125 voxels
    m = InstanceMap(voxel=0.1, min_count=1, reject_ratio=0.0)   # weighting alone, no rejection
    m.update([('chair', a, 0.8)], queries=('chair',))
    (y,) = m.instances
    assert abs(y.w - 100.0) < 1e-9                        # 125 x 0.8
    m.update([('chair', a, 0.2)], queries=('chair',))
    assert y.n == 250                                     # (4): evidence points as before
    assert abs(y.gamma - (100 * 0.8 + 25 * 0.2) / 125) < 1e-9          # 0.68 (eq. 5 as written: 0.5)
    weak = InstanceMap(voxel=0.1, min_count=1)
    weak.update([('chair', a, 0.1)], queries=('chair',))
    (z,) = weak.instances
    weak.update([('chair', a, 0.9)], queries=('chair',))
    assert abs(z.gamma - (12.5 * 0.1 + 112.5 * 0.9) / 125) < 1e-9      # 0.82 (as written: 0.5)


def test_score_weighted_negative_evidence():
    m = InstanceMap(voxel=0.1, min_conf=0.5, min_count=10, min_detections=1)
    a = box((0, 0, 0), (0.5, 0.5, 0.2))                   # 50 voxels
    m.update([('chair', a, 0.9)], queries=('chair',))
    (y,) = m.instances
    m.update([], queries=('chair',), visible=lambda pts: len(pts))      # 50 points of score 0
    assert y.n == 100 and abs(y.gamma - 50 * 0.9 / 100) < 1e-9          # (6) as written
    assert abs(y.w - 90.0) < 1e-9                                       # 45 + 50 x 45 / 50
    assert m.reliable('chair') == []


def test_weak_detections_alone_never_make_a_target():
    """A false positive seen often at 0.05-0.3 stays a tentative hypothesis."""
    m = InstanceMap(voxel=0.1, min_conf=0.5, min_count=10, min_detections=2)
    a = box((0, 0, 0), (0.5, 0.5, 0.5))
    for s in (0.25, 0.1, 0.3, 0.06, 0.2, 0.28):
        m.update([('chair', a, s)], queries=('chair',))
    (y,) = m.instances
    assert y.gamma < 0.3 and m.reliable('chair') == []


def test_inconsistent_low_score_is_rejected_as_transient_noise():
    """s_q < reject_ratio x gamma_y: skipped (no fusion, no count, no negative evidence)."""
    a = box((0, 0, 0), (0.5, 0.5, 0.5))
    m = InstanceMap(voxel=0.1, min_count=1, min_detections=1)
    m.update([('sofa', a, 0.8)], queries=('sofa',))
    (y,) = m.instances
    g, n = y.gamma, y.n
    m.update([('sofa', a, 0.2)], queries=('sofa',), visible=lambda pts: len(pts))   # 0.2 < 0.4
    assert y.gamma == g and y.n == n and y.n_pos == 1 and m.n_rejected == 1
    m.update([('sofa', a, 0.5)], queries=('sofa',))                                 # 0.5 >= 0.4
    assert y.n == 2 * n and y.gamma < g
    off = InstanceMap(voxel=0.1, min_count=1, reject_ratio=0.0)                     # off
    off.update([('sofa', a, 0.8)], queries=('sofa',))
    off.update([('sofa', a, 0.2)], queries=('sofa',))
    assert off.instances[0].n == 2 * n


def test_rejection_does_not_shield_a_weak_cluster():
    """A cluster that started weak takes weak detections (not below half of its own gamma)."""
    a = box((0, 0, 0), (0.5, 0.5, 0.5))
    m = InstanceMap(voxel=0.1, min_count=1)
    m.update([('sofa', a, 0.1)], queries=('sofa',))
    m.update([('sofa', a, 0.06)], queries=('sofa',))
    (y,) = m.instances
    assert y.n == 250 and m.n_rejected == 0
