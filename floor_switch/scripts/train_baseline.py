#!/usr/bin/env python3
"""p model baselines with scene-grouped 5-fold cross-validation (floor-switch plan, sec. 8).

Data: features (make_features.py) joined with labels (make_labels.py) at each decision time,
only times on a floor (in_band) of episodes in the train scenes of floor_switch_split.json.
Sample weight: 1 / (decision times of the episode), so every episode counts once; the positive
class gets pos_weight = (weighted negatives) / (weighted positives) of the training fold.

Models
  logreg  : linear on x (+ one-hot)
  mlp     : floor encoder Linear(15, H) + ReLU, concat one-hot, head Linear(H + 6, H) + ReLU +
            Linear(H, 1)   (the plan's layers; x has 15 numbers until the room histogram is added)
  single  : one feature as the score (sanity: does it rank at all)
Reports the episode-weighted AUROC per fold (mean, std) and writes out-of-fold predictions.
Run with torch available (e.g. `conda activate blip2`).
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from floor_switch.features import CATEGORIES, NAMES, one_hot  # noqa: E402

COUNTS = ('n_frontier', 'cand_pos', 'cand_neg', 'o_floors', 'o_exhausted', 'o_stair_floors')


def load(runs, feats, labels, split):
    sp = json.load(open(split))
    train_scenes = {s[:5] for s in sp['train']}
    eps = json.load(open(os.path.join(runs, 'episodes.json')))
    X, Y, W, G, C, E, T = [], [], [], [], [], [], []
    for e in eps:
        if e['id'][:5] not in train_scenes:
            continue
        fp, lp = os.path.join(feats, e['id'] + '.csv'), os.path.join(labels, e['id'] + '.csv')
        if not (os.path.exists(fp) and os.path.exists(lp)):
            continue
        lab = {round(float(r['t']), 2): r for r in csv.DictReader(open(lp))}
        rows = []
        for r in csv.DictReader(open(fp)):
            l = lab.get(round(float(r['t']), 2))
            if l is None or l['in_band'] != '1' or l['label'] == '':
                continue
            rows.append(([float(r[n]) for n in NAMES], int(l['label']), float(r['t'])))
        if not rows:
            continue
        w = 1.0 / len(rows)
        for x, y, t in rows:
            X.append(x); Y.append(y); W.append(w); G.append(e['id'][:5]); C.append(e['category'])
            E.append(e['id']); T.append(t)
    X = np.array(X)
    for n in COUNTS:
        j = NAMES.index(n)
        X[:, j] = np.log1p(X[:, j])
    return X, np.array(Y), np.array(W), np.array(G), np.array(C), np.array(E), np.array(T)


def auroc(score, y, w):
    """Weighted AUROC (probability that a random positive ranks above a random negative, ties 1/2)."""
    o = np.argsort(score, kind='mergesort')
    s, y, w = score[o], y[o], w[o]
    wn_below, auc, i = 0.0, 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        wp = w[i:j][y[i:j] == 1].sum()
        wn = w[i:j][y[i:j] == 0].sum()
        auc += wp * (wn_below + 0.5 * wn)
        wn_below += wn
        i = j
    P, N = w[y == 1].sum(), w[y == 0].sum()
    return auc / (P * N) if P > 0 and N > 0 else float('nan')


class MLP(torch.nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(h + k, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, x):
        if self.k:
            x, c = x[:, :-self.k], x[:, -self.k:]
            return self.head(torch.cat([self.enc(x), c], 1)).squeeze(1)
        return self.head(self.enc(x)).squeeze(1)


def fit(model, X, y, w, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    Xt, yt, wt = (torch.tensor(a, dtype=torch.float32) for a in (X, y, w))
    pw = wt[yt == 0].sum() / wt[yt == 1].sum()
    wt = wt * torch.where(yt == 1, pw, torch.tensor(1.0)) / wt.sum()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(Xt), yt, reduction='none') * wt).sum()
        loss.backward()
        opt.step()
    return float(pw)


def folds(groups, k, seed):
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    return [set(g[i::k]) for i in range(k)]


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--feats', default=H('~/objectnav_runs/floor_switch/features'))
    ap.add_argument('--labels', default=H('~/objectnav_runs/floor_switch/labels'))
    ap.add_argument('--split', default=H('~/objectnav_runs/floor_switch_split.json'))
    ap.add_argument('--out', default=H('~/objectnav_runs/floor_switch/baseline'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=400, help='full-batch Adam steps, logistic regression')
    ap.add_argument('--mlp-epochs', type=int, default=400)
    ap.add_argument('--mlp-lr', type=float, default=1e-2)
    ap.add_argument('--mlp-wd', type=float, default=1e-4)
    args = ap.parse_args()

    X, Y, W, G, C, E, T = load(args.runs, args.feats, args.labels, args.split)
    OH = np.stack([one_hot(c) for c in C])
    print(f'{len(set(E))} episodes, {len(set(G))} scenes, {len(Y)} decision times, '
          f'y=1 {Y.mean():.3f} (episode-weighted {W[Y == 1].sum() / W.sum():.3f})')
    fs = folds(G, args.folds, args.seed)
    os.makedirs(args.out, exist_ok=True)

    configs = [('logreg x', 'lin', False, 0), ('logreg x+onehot', 'lin', True, 0),
               ('mlp32 x+onehot', 'mlp', True, 32), ('mlp64 x+onehot', 'mlp', True, 64),
               ('mlp64 x', 'mlp', False, 64)]
    results = {}
    for name, kind, oh, h in configs:
        oof = np.full(len(Y), np.nan)
        per = []
        for fi, test in enumerate(fs):
            te = np.array([g in test for g in G])
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xn = (X - mu) / sd
            Z = np.concatenate([Xn, OH], 1) if oh else Xn
            k = OH.shape[1] if oh else 0
            model = torch.nn.Linear(Z.shape[1], 1) if kind == 'lin' else MLP(X.shape[1], h, k)
            if kind == 'lin':
                lin = model
                model = torch.nn.Sequential(lin, torch.nn.Flatten(0))
            if kind == 'lin':
                fit(model, Z[tr], Y[tr], W[tr], args.epochs, 1e-2, 1e-4, args.seed + fi)
            else:
                fit(model, Z[tr], Y[tr], W[tr], args.mlp_epochs, args.mlp_lr, args.mlp_wd, args.seed + fi)
            with torch.no_grad():
                oof[te] = model(torch.tensor(Z[te], dtype=torch.float32)).numpy()
            per.append(auroc(oof[te], Y[te], W[te]))
        results[name] = (np.mean(per), np.std(per), auroc(oof, Y, W), oof)
        print(f'{name:18s} AUROC fold mean {np.mean(per):.3f} +- {np.std(per):.3f}  pooled {results[name][2]:.3f}  '
              f'folds {" ".join(f"{a:.3f}" for a in per)}', flush=True)

    print('single features (pooled AUROC; < 0.5 means the feature points to "not on this floor"):')
    for j, n in enumerate(NAMES):
        print(f'  {n:15s} {auroc(X[:, j], Y, W):.3f}')

    best = 'mlp64 x+onehot'
    print(f'{best} pooled AUROC per category / A-B:')
    eps = {e['id']: e for e in json.load(open(os.path.join(args.runs, 'episodes.json')))}
    AB = np.array(['A' if eps[e]['goal_on_start_floor'] else 'B' for e in E])
    oof = results[best][3]
    for c in CATEGORIES:
        m = C == c
        print(f'  {c:11s} n_ep {len(set(E[m])):3d}  {auroc(oof[m], Y[m], W[m]):.3f}')
    for ab in 'AB':
        m = AB == ab
        print(f'  {ab} episodes   n_ep {len(set(E[m])):3d}  y=1 {W[m][Y[m] == 1].sum() / W[m].sum():.3f}  '
              f'AUROC {auroc(oof[m], Y[m], W[m]):.3f}')
    with open(os.path.join(args.out, 'oof.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        names = list(results)
        wr.writerow(['id', 't', 'category', 'ab', 'y', 'w'] + names)
        for i in range(len(Y)):
            wr.writerow([E[i], T[i], C[i], AB[i], Y[i], f'{W[i]:.6g}'] + [f'{results[n][3][i]:.4f}' for n in names])
    json.dump({n: dict(mean=r[0], std=r[1], pooled=r[2]) for n, r in results.items()},
              open(os.path.join(args.out, 'auroc.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
