#!/usr/bin/env python3
"""Per-category calibration of the p model (floor-switch plan, sec. 8.5) and its reliability check.

Takes the out-of-fold logits of one model of train_baseline.py (trained with --cat-weights) and
applies, per fold and category c, the constant
    logit(p_cal) = logit(p_model) - log(pos_weight) - logit(q_c) + logit(pi_c)
  pos_weight: (weighted negatives) / (weighted positives) of the fold's training samples, the
              shift weighted BCE puts into the logits;
  q_c       : y = 1 share of category c in the training samples, with the training weights
              without pos_weight (1 / episode length x category episode weight);
  pi_c      : y = 1 share of category c with the real-mix weights (1 / episode length x
              pi_c / q_c of the A/B mix, all train scenes).
The constants of a fold use its training scenes only. Reliability (episode-weighted with the real
mix): 10 bins of p, mean p vs y = 1 share, ECE and Brier, overall, per category and within A / B.
Compared with no correction and with Platt scaling (a, b per category one-hot, fitted on the
other folds' predictions). Writes <out>/calibration.json (the constants on all train scenes, for the
online node), <out>/oof_calibrated.csv (out-of-fold p per decision time) and <out>/reliability.csv.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from train_baseline import category_weights, folds, load  # noqa: E402
from floor_switch.features import CATEGORIES  # noqa: E402

logit = lambda p: np.log(p / (1.0 - p))
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))


def share(y, w, m):
    return float(w[m & (y == 1)].sum() / w[m].sum())


def constants(Y, Wtr, Wreal, C, m):
    """pos_weight and the per-category correction from the samples m (training scenes)."""
    pw = Wtr[m & (Y == 0)].sum() / Wtr[m & (Y == 1)].sum()
    corr = {}
    for c in CATEGORIES:
        mc = m & (C == c)
        q, pi = share(Y, Wtr, mc), share(Y, Wreal, mc)
        q, pi = np.clip([q, pi], 1e-3, 1 - 1e-3)
        corr[c] = float(-np.log(pw) - logit(q) + logit(pi))
    return float(pw), corr


def reliability(p, y, w, bins=10):
    e = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (p >= e[i]) & ((p < e[i + 1]) if i < bins - 1 else (p <= 1))
        if w[m].sum() <= 0:
            continue
        mp, my, ww = np.average(p[m], weights=w[m]), np.average(y[m], weights=w[m]), w[m].sum() / w.sum()
        ece += ww * abs(mp - my)
        rows.append((e[i], e[i + 1], ww, mp, my))
    brier = float(np.average((p - y) ** 2, weights=w))
    return ece, brier, rows


def platt(z, oh, y, w, steps=500):
    """Weighted logistic fit of y on (z, one-hot): one slope, one offset per category."""
    X = torch.tensor(np.concatenate([z[:, None], oh], 1), dtype=torch.float32)
    yt, wt = torch.tensor(y, dtype=torch.float32), torch.tensor(w / w.sum(), dtype=torch.float32)
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(oh.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=steps)

    def closure():
        opt.zero_grad()
        out = a * X[:, 0] + X[:, 1:] @ b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(out, yt, reduction='none') * wt).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.detach()), b.detach().numpy()


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--feats', default=H('~/objectnav_runs/floor_switch/features'))
    ap.add_argument('--labels', default=H('~/objectnav_runs/floor_switch/labels'))
    ap.add_argument('--rooms', default=H('~/objectnav_runs/floor_switch/rooms/m0.075_T0.1'))
    ap.add_argument('--split', default=H('~/objectnav_runs/floor_switch_split.json'))
    ap.add_argument('--candidates', default=H('~/objectnav_runs/collect_v01/candidates.json'))
    ap.add_argument('--oof', default=H('~/objectnav_runs/floor_switch/p_model/catw/oof.csv'))
    ap.add_argument('--model', default='mlp32 x+onehot')
    ap.add_argument('--out', default=H('~/objectnav_runs/floor_switch/p_model/catw'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    X, Y, W, G, C, E, T, AB = load(args.runs, args.feats, args.labels, args.split, args.rooms)
    cands = json.load(open(args.candidates))
    oof = {(r['id'], round(float(r['t']), 2)): float(r[args.model]) for r in csv.DictReader(open(args.oof))}
    Z = np.array([oof[(e, round(float(t), 2))] for e, t in zip(E, T)])
    CWall, _, _ = category_weights(cands, set(G), C, AB, E, G)
    Wreal = W * CWall
    OH = np.array([[c == k for k in CATEGORIES] for c in C], dtype=np.float64)

    fs = folds(G, args.folds, args.seed)
    p_raw = sigmoid(Z)
    p_formula = np.zeros_like(Z)
    p_platt = np.zeros_like(Z)
    per_fold = []
    for fi, test in enumerate(fs):
        te = np.isin(G, list(test))
        tr = ~te
        cw, _, _ = category_weights(cands, set(G[tr]), C, AB, E, G)
        Wtr = W * cw
        pw, corr = constants(Y, Wtr, Wreal, C, tr)
        per_fold.append(dict(pos_weight=pw, correction=corr))
        p_formula[te] = sigmoid(Z[te] + np.array([corr[c] for c in C[te]]))
        a, b = platt(Z[tr], OH[tr], Y[tr], Wreal[tr])       # the other folds' out-of-fold logits
        p_platt[te] = sigmoid(a * Z[te] + OH[te] @ b)
        per_fold[-1]['platt'] = dict(slope=a, offset=dict(zip(CATEGORIES, map(float, b))))

    print(f'model {args.model}, {len(set(E))} episodes, real-mix weights')
    print('fold constants: pos_weight and correction per category')
    for fi, f in enumerate(per_fold):
        print(f'  fold {fi + 1}: pos_weight {f["pos_weight"]:.3f} (-log {-np.log(f["pos_weight"]):+.3f})  '
              + ' '.join(f'{c[:5]} {v:+.3f}' for c, v in f['correction'].items()))
    rel_rows = []
    print(f'{"":22s} {"ECE":>6s} {"Brier":>6s}   ' + '  '.join(f'{c[:6]:>11s}' for c in CATEGORIES)
          + '   within A / B (mean p vs y=1)')
    for name, p in (('no correction', p_raw), ('formula (sec. 8.5)', p_formula), ('Platt (CV)', p_platt)):
        ece, brier, rows = reliability(p, Y, Wreal)
        cat = []
        for c in CATEGORIES:
            m = C == c
            cat.append(f'{reliability(p[m], Y[m], Wreal[m])[0]:.3f}/{np.average(p[m], weights=Wreal[m]) - np.average(Y[m], weights=Wreal[m]):+.2f}')
        ab = []
        for k in 'AB':
            m = AB == k
            ab.append(f'{k} {np.average(p[m], weights=Wreal[m]):.3f} vs {np.average(Y[m], weights=Wreal[m]):.3f}')
        print(f'{name:22s} {ece:6.3f} {brier:6.3f}   ' + '  '.join(f'{s:>11s}' for s in cat) + '   ' + ', '.join(ab))
        for lo, hi, ww, mp, my in rows:
            rel_rows.append([name, f'{lo:.1f}', f'{hi:.1f}', f'{ww:.4f}', f'{mp:.4f}', f'{my:.4f}'])
    print('  (per category: ECE / mean p - y=1 share)')
    print('reliability bins (share of weight, mean p, y=1 share):')
    for name in ('no correction', 'formula (sec. 8.5)', 'Platt (CV)'):
        print(f'  {name:20s} ' + '  '.join(f'[{r[1]}-{r[2]}] {float(r[3]):.2f} {float(r[4]):.2f}/{float(r[5]):.2f}'
                                         for r in rel_rows if r[0] == name))

    # constants on all train scenes (for the online node)
    cw, _, _ = category_weights(cands, set(G), C, AB, E, G)
    pw, corr = constants(Y, W * cw, Wreal, C, np.ones(len(Y), dtype=bool))
    os.makedirs(args.out, exist_ok=True)
    a, b = platt(Z, OH, Y, Wreal)
    print(f'Platt on all out-of-fold logits: slope {a:.3f}, offset ' + ' '.join(f'{c} {v:+.3f}' for c, v in zip(CATEGORIES, b)))
    print('Platt slopes per fold: ' + ' '.join(f'{f["platt"]["slope"]:.3f}' for f in per_fold))
    json.dump(dict(model=args.model, pos_weight=pw, correction=corr, per_fold=per_fold,
                   platt=dict(slope=a, offset=dict(zip(CATEGORIES, map(float, b))),
                              formula='p = sigmoid(slope * logit_model + offset[category])'),
                   formula='logit(p_cal) = logit(p_model) - log(pos_weight) - logit(q_c) + logit(pi_c)'),
              open(os.path.join(args.out, 'calibration.json'), 'w'), indent=1)
    with open(os.path.join(args.out, 'oof_calibrated.csv'), 'w', newline='') as f:
        w = csv.writer(f)                     # out-of-fold p per decision time (Platt fitted on the other folds)
        w.writerow(['id', 't', 'category', 'ab', 'y', 'w_real', 'p_raw', 'p_formula', 'p_platt'])
        for i in range(len(Y)):
            w.writerow([E[i], T[i], C[i], AB[i], Y[i], f'{Wreal[i]:.6g}', f'{p_raw[i]:.4f}',
                        f'{p_formula[i]:.4f}', f'{p_platt[i]:.4f}'])
    with open(os.path.join(args.out, 'reliability.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'p_lo', 'p_hi', 'weight_share', 'mean_p', 'y1_share'])
        w.writerows(rel_rows)
    print(f'all train scenes: pos_weight {pw:.3f}, correction ' + ' '.join(f'{c} {v:+.3f}' for c, v in corr.items()))


if __name__ == '__main__':
    main()
