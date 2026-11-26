"""The p model in numpy for the online node (no torch in the ROS python).

online_model.json (scripts/export_model.py): the five MLP32 fold models of the CV (category
episode weights), each with the input normalisation of its training scenes; the logit is their
mean. The Platt constants (calibration.json, fitted on these models' out-of-fold logits) and the
rule constants (decision_constants.json, rule_selection.json).

Input: the 15 floor features (features.NAMES) + the 9-bin room histogram (rooms.ROOMS);
log1p on the count features (COUNTS), then (x - mu) / sd; the target one-hot goes to the head.
  logit_k = W3 relu(W2 [relu(W1 x_k + b1), one_hot] + b2) + b3   (x_k: fold k's normalisation)
  logit   = mean over the fold models
  p     = sigmoid(slope * logit + offset[category])
"""
import json
import math

import numpy as np

from floor_switch.features import NAMES, one_hot

COUNTS = ('n_frontier', 'cand_pos', 'cand_neg', 'o_floors', 'o_exhausted', 'o_stair_floors')


def preprocess(x):
    """log1p of the count features (train_baseline.load); x: (..., 15 or 24)."""
    x = np.array(x, dtype=np.float64)
    for n in COUNTS:
        j = NAMES.index(n)
        x[..., j] = np.log1p(x[..., j])
    return x


class _MLP:
    def __init__(self, m):
        w = {k: np.asarray(v, dtype=np.float64) for k, v in m['weights'].items()}
        self.W1, self.b1 = w['enc.0.weight'], w['enc.0.bias']
        self.W2, self.b2 = w['head.0.weight'], w['head.0.bias']
        self.W3, self.b3 = w['head.2.weight'], w['head.2.bias']
        self.mu, self.sd = np.asarray(m['mu']), np.asarray(m['sd'])

    def logit(self, xp, oh):
        z = (xp - self.mu) / self.sd
        h = np.maximum(self.W1 @ z + self.b1, 0.0)
        h = np.maximum(self.W2 @ np.concatenate([h, oh]) + self.b2, 0.0)
        return float((self.W3 @ h + self.b3)[0])


class PModel:
    def __init__(self, path):
        d = json.load(open(path))
        self.meta = d
        self.members = [_MLP(m) for m in d['members']]
        self.slope = float(d['platt']['slope'])
        self.offset = {c: float(v) for c, v in d['platt']['offset'].items()}
        self.n_in = len(self.members[0].mu)

    def logit(self, x, target):
        """Mean logit of the fold models for one sample: x the raw features (15 + 9 rooms)."""
        xp, oh = preprocess(x), one_hot(target)
        return float(np.mean([m.logit(xp, oh) for m in self.members]))

    def p(self, x, target):
        """Calibrated p (target on the current floor) and the mean logit."""
        z = self.logit(x, target)
        a = self.slope * z + self.offset.get(target, 0.0)
        return 1.0 / (1.0 + math.exp(-a)), z
