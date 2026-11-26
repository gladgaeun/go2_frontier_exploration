#!/usr/bin/env python3
"""Choose X and tau of the v0 switch rule offline (floor-switch plan, sec. 9.1 and 9.4).

Replays the rule on the collected train episodes with the out-of-fold calibrated p
(calibrate.py, Platt, each fold's p from a model that did not see that scene):
  - a floor visit: a run of decision times on one floor of the robot's estimate (FloorTracker on
    z_r, in band); the p history is cleared at its start and whenever target tracking starts or
    ends, and the rule does not fire while a target is tracked;
  - decision times: the frontier_pose requests and every 10 s;
  - fire when the last X p (1 s each) are all below tau (X records at least) and the cost guard
    passes: actions left >= actions/m x distance to the nearest known stair entrance of this floor
    + climb p75 + margin (decision_constants.json); no known stairs: no switch.
Outcome of a visit (real A/B mix weights per episode):
  - y = 0 at the firing time (target not on this floor): saved actions = the actions the robot
    still spent on that floor after it;
  - y = 1 at the firing time: a wrong switch; cost = a round trip 2 x (stairs + climb actions).
Net = saved - wrong-switch cost, per episode. Also the B share 15-30 % sensitivity of the best
(X, tau). Writes <out>/rule_selection.json.
"""
import argparse
import bisect
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from floor_switch.features import FloorTracker  # noqa: E402


def read_jsonl(path, kind):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('type') == kind:
                out.append(r)
    return out


def cum_actions(path, step=0.25, turn=math.radians(30.0)):
    a = np.loadtxt(path, ndmin=2)
    d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 3]))
    r = np.abs((np.diff(a[:, 4]) + math.pi) % (2.0 * math.pi) - math.pi)
    return list(a[:, 0]), np.concatenate([[0.0], np.cumsum(d / step + r / turn)])


def stair_distance(state):
    """Distance (m, straight line) from the robot to the nearest known (reliable) stair entrance
    of its floor (foot on this floor going up, or reaching down from it); None if none."""
    z, rob = state.get('z_r'), state.get('robot')
    if z is None or not rob or rob[0] is None:
        return None
    best = None
    for lm in state.get('landmarks') or []:
        if len(lm) > 7 and lm[7] < 0.5:
            continue
        if abs(lm[2] - z) <= 0.4 or (lm[2] < z - 0.5 and lm[5] >= z - 1.0):
            d = math.hypot(lm[0] - rob[0], lm[1] - rob[1])
            best = d if best is None or d < best else best
    return best


def episode_visits(run_dir, rows):
    """Floor visits of one episode: lists of dicts per decision time (labelled, in band)."""
    rd = run_dir
    fg = read_jsonl(os.path.join(rd, 'record', 'frontier_goal.jsonl'), 'state')
    req = [r['t'] for r in read_jsonl(os.path.join(rd, 'record', 'frontier_goal.jsonl'), 'request')]
    vm = read_jsonl(os.path.join(rd, 'record', 'voxel_mapper.jsonl'), 'state')
    vm_t = [r['t'] for r in vm]
    tt, acts = cum_actions(os.path.join(rd, 'trajectory.txt'))
    by_t = {round(float(r['t']), 2): r for r in rows}
    ft = FloorTracker()
    visits, cur, last_floor = [], None, None
    for s in fg:
        t = round(float(s['t']), 2)
        z = s.get('z_r')
        j = bisect.bisect_right(vm_t, t + 1e-6) - 1
        climbing = bool(vm[j].get('climbing')) if j >= 0 else False
        k = ft.update(z, climbing) if z is not None else None
        r = by_t.get(t)
        if r is None or k is None:                       # stairs / not labelled: the visit ends
            last_floor = None
            continue
        if k != last_floor:
            cur = []
            visits.append(cur)
            last_floor = k
        a = float(acts[max(0, bisect.bisect_right(tt, t + 1e-6) - 1)])
        cur.append(dict(t=t, p=float(r['p_platt']), y=int(r['y']), actions=a,
                        tracking=bool(s.get('tracking_target')), stairs=stair_distance(s)))
    return visits, sorted(req)


def simulate(visits, req, X, tau, k_floor, guard_fixed, max_actions, round_trip):
    """(saved actions on y=0 visits, wrong switches, wrong-switch cost, y=1 visits, y=0 visits)."""
    saved = wrong = cost = 0.0
    n1 = n0 = 0
    for v in visits:
        if not v:
            continue
        y_visit = v[0]['y']
        n1 += y_visit == 1
        n0 += y_visit == 0
        hist = []
        fired = None
        next_tick = v[0]['t'] + 10.0
        ri = bisect.bisect_left(req, v[0]['t'])
        tracking = False
        for d in v:
            if d['tracking'] != tracking:                 # tracking starts or ends: clear
                hist, tracking = [], d['tracking']
            if tracking:
                continue
            hist.append(d['p'])
            decide = False
            if d['t'] >= next_tick:
                decide, next_tick = True, d['t'] + 10.0
            while ri < len(req) and req[ri] <= d['t']:
                decide, ri = True, ri + 1
            if not decide or len(hist) < X or max(hist[-X:]) >= tau or d['stairs'] is None:
                continue
            need = k_floor * d['stairs'] + guard_fixed
            if max_actions - d['actions'] < need:
                continue
            fired = d
            break
        if fired is None:
            continue
        if fired['y'] == 0:
            saved += v[-1]['actions'] - fired['actions']
        else:
            wrong += 1
            cost += round_trip(fired['stairs'])
    return saved, wrong, cost, n1, n0


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--model-dir', default=H('~/objectnav_runs/floor_switch/p_model/catw'))
    ap.add_argument('--max-actions', type=float, default=500.0)
    ap.add_argument('--X', default='5,10,20,30,60')
    ap.add_argument('--tau', default='0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8')
    ap.add_argument('--b-shares', default='0.15,0.2,0.25,0.3')
    args = ap.parse_args()

    dc = json.load(open(os.path.join(args.model_dir, 'decision_constants.json')))
    k_floor = dc['actions_per_metre_floor']
    climb = dc['climb_actions']['changed_floor']['p75']
    margin = dc['margin']['km_median'] or dc['margin']['success_p75']
    guard_fixed = climb + margin
    round_trip = lambda dist: 2.0 * (k_floor * dist + climb)
    print(f'cost guard: actions left >= {k_floor:.2f} x stair distance + {climb:.0f} (climb p75) + {margin:.0f} (margin)')

    rows = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(args.model_dir, 'oof_calibrated.csv'))):
        rows[r['id']].append(r)
    eps = {}
    for eid, rs in rows.items():
        visits, req = episode_visits(os.path.join(args.runs, eid), rs)
        ew = sum(float(r['w_real']) for r in rs)            # episode weight: the real-mix A/B weight
        eps[eid] = dict(visits=visits, req=req, ew=ew, ab=rs[0]['ab'])
    b0 = sum(e['ew'] for e in eps.values() if e['ab'] == 'B') / sum(e['ew'] for e in eps.values())
    n_vis = sum(len(e['visits']) for e in eps.values())
    print(f'{len(eps)} episodes, {n_vis} floor visits, real-mix B share {b0:.3f}')

    Xs = [int(x) for x in args.X.split(',')]
    taus = [float(x) for x in args.tau.split(',')]
    sims = {}
    for X in Xs:
        for tau in taus:
            sims[(X, tau)] = {eid: simulate(e['visits'], e['req'], X, tau, k_floor, guard_fixed,
                                            args.max_actions, round_trip) for eid, e in eps.items()}

    def score(res, b):
        wA, wB = (1 - b) / (1 - b0), b / b0
        tot = saved = wrong = cost = n1 = n0 = 0.0
        for eid, (s, w, c, v1, v0) in res.items():
            ew = eps[eid]['ew'] * (wB if eps[eid]['ab'] == 'B' else wA)
            tot += ew; saved += ew * s; wrong += ew * w; cost += ew * c; n1 += ew * v1; n0 += ew * v0
        return dict(saved=saved / tot, wrong_rate=wrong / n1 if n1 else 0.0, cost=cost / tot,
                    net=(saved - cost) / tot)

    base = {k: score(v, b0) for k, v in sims.items()}
    print(f'\nper episode, real mix (B {b0:.2f}): saved actions / wrong-switch rate (of visits on the target floor) / net')
    print('  X \\ tau ' + ''.join(f'{t:>17.2f}' for t in taus))
    for X in Xs:
        print(f'  {X:3d} s   ' + ''.join(f'{base[(X, t)]["saved"]:6.1f}/{100 * base[(X, t)]["wrong_rate"]:4.1f}%/{base[(X, t)]["net"]:+5.1f}'.rjust(17)
                                      for t in taus))
    best = max(base, key=lambda k: base[k]['net'])
    print(f'\nbest net at the real mix: X {best[0]} s, tau {best[1]:.2f}: {base[best]}')
    sens = {}
    print('B share sensitivity (best X, tau by net):')
    for b in [float(x) for x in args.b_shares.split(',')]:
        sc = {k: score(v, b) for k, v in sims.items()}
        bk = max(sc, key=lambda k: sc[k]['net'])
        sens[b] = dict(X=bk[0], tau=bk[1], **sc[bk], chosen_at_this_share=score(sims[best], b))
        print(f'  B {b:.2f}: best X {bk[0]:3d} s tau {bk[1]:.2f}  net {sc[bk]["net"]:+.1f}  | '
              f'(X {best[0]}, tau {best[1]:.2f}) net {score(sims[best], b)["net"]:+.1f}, '
              f'wrong {100 * score(sims[best], b)["wrong_rate"]:.1f}%')
    json.dump(dict(cost_guard=dict(actions_per_metre=k_floor, climb_p75=climb, margin=margin),
                   real_b_share=b0, best=dict(X=best[0], tau=best[1], **base[best]),
                   grid={f'X{k[0]}_tau{k[1]}': v for k, v in base.items()},
                   sensitivity={str(b): v for b, v in sens.items()}),
              open(os.path.join(args.model_dir, 'rule_selection.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
