#!/usr/bin/env python3
"""v0 p model for the online node: <model-dir>/online_model.json.

The five fold models of the scene-grouped CV (train_baseline.py --cat-weights, MLP32 x 24 +
one-hot), rebuilt with the same folds, weights, normalisation and seeds, and averaged: the online
logit is the mean of the five, and the Platt constants (calibration.json) were fitted on these
models' out-of-fold logits, so model and calibration match. Each fold model is checked against
its out-of-fold logits in oof.csv. Saved with each its input normalisation, the Platt constants
and the rule constants (decision_constants.json, rule_selection.json); the numpy forward
(floor_switch.model.PModel) is checked against torch.
Check on sim_dev (not used for any choice): AUROC (episode / real-mix weights), within B, and the
ECE of the calibrated p.
Run with torch available (e.g. `conda activate blip2`).
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
from train_baseline import MLP, auroc, category_weights, fit, folds, load  # noqa: E402
from calibrate import reliability  # noqa: E402
from floor_switch.features import NAMES, one_hot  # noqa: E402
from floor_switch.model import PModel  # noqa: E402
from floor_switch.rooms import ROOMS  # noqa: E402


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--feats', default=H('~/objectnav_runs/floor_switch/features'))
    ap.add_argument('--labels', default=H('~/objectnav_runs/floor_switch/labels'))
    ap.add_argument('--rooms', default=H('~/objectnav_runs/floor_switch/rooms/m0.075_T0.1'))
    ap.add_argument('--split', default=H('~/objectnav_runs/floor_switch_split.json'))
    ap.add_argument('--candidates', default=H('~/objectnav_runs/collect_v01/candidates.json'))
    ap.add_argument('--model-dir', default=H('~/objectnav_runs/floor_switch/p_model/catw'))
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=200, help='the CV run of p_model/catw: --mlp-epochs 200 --mlp-lr 3e-3 --mlp-wd 1e-3')
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--room-margin', type=float, default=0.075)
    ap.add_argument('--room-T', type=float, default=0.1)
    args = ap.parse_args()

    torch.set_num_threads(4)
    X, Y, W, G, C, E, T, AB = load(args.runs, args.feats, args.labels, args.split, args.rooms)
    cands = json.load(open(args.candidates))
    OH = np.stack([one_hot(c) for c in C])
    oof = {(r['id'], round(float(r['t']), 2)): float(r['mlp32 x+onehot'])
           for r in csv.DictReader(open(os.path.join(args.model_dir, 'oof.csv')))}
    Zoof = np.array([oof[(e, round(float(t), 2))] for e, t in zip(E, T)])
    members = []
    for fi, test in enumerate(folds(G, args.folds, args.seed)):
        te = np.isin(G, list(test))
        tr = ~te
        cw, _, _ = category_weights(cands, set(G[tr]), C, AB, E, G)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = np.concatenate([(X - mu) / sd, OH], 1)
        # train_baseline builds each model before fit() seeds: its initial weights come from the
        # seed of the fit before (logreg x+onehot fold 5 for the first fold, the fold before after)
        torch.manual_seed(args.seed + (args.folds - 1 if fi == 0 else fi - 1))
        model = MLP(X.shape[1], args.hidden, OH.shape[1])
        pw = fit(model, Z[tr], Y[tr], (W * cw)[tr], args.epochs, args.lr, args.wd, args.seed + fi)
        with torch.no_grad():
            z = model(torch.tensor(Z[te], dtype=torch.float32)).numpy()
        dz = np.abs(z - Zoof[te]).max()
        print(f'fold {fi + 1}: {len(set(G[tr]))} training scenes, pos_weight {pw:.3f}, '
              f'out-of-fold logits vs oof.csv max |diff| {dz:.4f}')
        if dz > 0.01:
            raise SystemExit('fold model differs from the CV run (oof.csv); not written')
        members.append(dict(mu=mu.tolist(), sd=sd.tolist(), test_scenes=sorted(test),
                            weights={k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}))

    cal = json.load(open(os.path.join(args.model_dir, 'calibration.json')))
    dc = json.load(open(os.path.join(args.model_dir, 'decision_constants.json')))
    rs = json.load(open(os.path.join(args.model_dir, 'rule_selection.json')))
    out = dict(
        model=f'mlp{args.hidden} x+onehot, mean logit of the {args.folds} CV fold models, category episode weights',
        inputs=list(NAMES) + [r.replace(' ', '_') for r in ROOMS], log1p='floor_switch.model.COUNTS',
        rooms=dict(margin=args.room_margin, T=args.room_T, res=0.05),
        members=members,
        platt=cal['platt'],
        rule=dict(X=rs['best']['X'], tau=rs['best']['tau'],
                  actions_per_metre=dc['actions_per_metre_floor'],
                  climb_p75=dc['climb_actions']['changed_floor']['p75'],
                  margin=dc['margin']['km_median'] or dc['margin']['success_p75']),
        train=dict(episodes=len(set(E)), scenes=len(set(G)), samples=len(Y), folds=args.folds,
                   epochs=args.epochs, lr=args.lr, wd=args.wd, seed=args.seed))
    path = os.path.join(args.model_dir, 'online_model.json')
    json.dump(out, open(path, 'w'))
    print(f'wrote {path}')

    # numpy forward = the mean of the torch fold models
    pm = PModel(path)
    raw = np.array(X)
    from floor_switch.model import COUNTS
    for n in COUNTS:                     # load() applied log1p already; PModel applies it itself
        j = NAMES.index(n)
        raw[:, j] = np.expm1(raw[:, j])
    zt = np.zeros(len(Y))
    for m in members:
        net = MLP(X.shape[1], args.hidden, OH.shape[1])
        net.load_state_dict({k: torch.tensor(v) for k, v in m['weights'].items()})
        with torch.no_grad():
            zt += net(torch.tensor(np.concatenate([(X - np.array(m['mu'])) / np.array(m['sd']), OH], 1),
                                   dtype=torch.float32)).numpy() / len(members)
    zn = np.array([pm.logit(raw[i], C[i]) for i in range(len(Y))])
    print(f'numpy vs torch mean logit: max |diff| {np.abs(zn - zt).max():.2e}')

    # sim_dev check
    Xd, Yd, Wd, Gd, Cd, Ed, Td, ABd = load(args.runs, args.feats, args.labels, args.split, args.rooms,
                                             which='sim_dev')
    raw = np.array(Xd)
    for n in COUNTS:
        j = NAMES.index(n)
        raw[:, j] = np.expm1(raw[:, j])
    pd = np.array([pm.p(raw[i], Cd[i])[0] for i in range(len(Yd))])
    cwd, _, _ = category_weights(cands, set(Gd), Cd, ABd, Ed, Gd)
    Wr = Wd * cwd
    b = ABd == 'B'
    ece, brier, _ = reliability(pd, Yd, Wr)
    print(f'sim_dev: {len(set(Ed))} episodes ({len(set(Ed[b]))} B), {len(set(Gd))} scenes, {len(Yd)} decision times')
    print(f'  AUROC {auroc(pd, Yd, Wd):.3f} (episode weights), {auroc(pd, Yd, Wr):.3f} (real mix), '
          f'within B {auroc(pd[b], Yd[b], Wd[b]):.3f}; ECE {ece:.3f}, Brier {brier:.3f}')
    for k in 'AB':
        m = ABd == k
        print(f'  {k}: mean p {np.average(pd[m], weights=Wr[m]):.3f} vs y=1 share {np.average(Yd[m], weights=Wr[m]):.3f}')


if __name__ == '__main__':
    main()
