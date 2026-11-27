#!/usr/bin/env python3
"""Floor switches of runs with floor_switch:=on, next to a reference run without the module.

Per finished episode (results.csv of score_episodes.py when there, else result.json):
  A/B, success and actions (on vs the reference run of the same episode id), and each switch:
  time, p, the floor it left (nearest level of the episode, Habitat y = map z + start y),
  correct = the target is not on that floor (episode goal_floors), and how the climb ended
  (launch.log: went up / down to a new floor, did not reach one, called off).
Then the totals: SR and mean actions on vs reference (A / B), switches correct / wrong, and the
episodes that failed after a wrong switch.
"""
import argparse
import csv
import json
import os
import re


def result(run_dir, scored):
    rid = os.path.basename(run_dir)
    rp = os.path.join(run_dir, 'result.json')
    if not os.path.exists(rp):
        return None
    r = json.load(open(rp))
    s = scored.get(rid)
    return dict(success=None if s is None else s['success'] == 'True', actions=float(r.get('actions', 0.0)),
                status=r.get('status'))


def switches(run_dir, ep):
    out = []
    fg = os.path.join(run_dir, 'record', 'frontier_goal.jsonl')
    if not os.path.exists(fg):
        return out
    for line in open(fg):
        r = json.loads(line)
        if r.get('type') != 'switch':
            continue
        y = r['z_r'] + ep['start'][1]
        lv = min(range(len(ep['levels'])), key=lambda i: abs(ep['levels'][i] - y))
        st = r.get('state') or {}
        out.append(dict(t=r['t'], p=st.get('p'), level=lv, start=lv == ep['start_floor'],
                        correct=lv not in ep['goal_floors']))
    # how each switch climb ended (launch.log order)
    log = os.path.join(run_dir, 'launch.log')
    ends = []
    if os.path.exists(log):
        cur = None
        for line in open(log, errors='ignore'):
            if '[stairs] floor switch (' in line:
                cur = len(ends)
                ends.append('?')
            elif cur is not None:
                m = re.search(r'\[stairs\] went (up|down): z_r ([-0-9.]+) -> ([-0-9.]+)', line)
                if m:
                    ends[cur], cur = f'{m.group(1)} to {float(m.group(3)):.1f}', None
                elif 'did not reach a new floor' in line:
                    ends[cur], cur = 'no new floor', None
                elif 'calling the climb off' in line:
                    ends[cur], cur = 'called off', None
    for s, e in zip(out, ends):
        s['end'] = e
    return out


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/fs_on_simdev_v0'))
    ap.add_argument('--ref', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--last', type=int, default=0, help='list only the last N finished episodes (0: all)')
    args = ap.parse_args()
    eps = json.load(open(os.path.join(args.runs, 'episodes.json')))
    load = lambda d: ({r['id']: r for r in csv.DictReader(open(os.path.join(d, 'results.csv')))}
                      if os.path.exists(os.path.join(d, 'results.csv')) else {})
    sc_on, sc_ref = load(args.runs), load(args.ref)
    rows = []
    for ep in eps:
        on = result(os.path.join(args.runs, ep['id']), sc_on)
        if on is None:
            continue
        ref = result(os.path.join(args.ref, ep['id']), sc_ref)
        rows.append((ep, on, ref, switches(os.path.join(args.runs, ep['id']), ep)))
    fmt = lambda s: '-' if s is None else ('O' if s else 'X')
    show = rows[-args.last:] if args.last else rows
    print(f'{"episode":26s} {"":2s} {"on":>3s} {"ref":>3s} {"act on":>7s} {"ref":>6s}  switches (t s, p, left floor: correct?, climb)')
    for ep, on, ref, sw in show:
        ab = 'A' if ep['goal_on_start_floor'] else 'B'
        s = '; '.join(f'{x["t"]:.0f}s p {x["p"]:.2f} L{x["level"]}{"(start)" if x["start"] else ""} '
                      f'{"right" if x["correct"] else "WRONG"}, {x.get("end", "?")}' if x['p'] is not None else
                      f'{x["t"]:.0f}s L{x["level"]} {"right" if x["correct"] else "WRONG"}, {x.get("end", "?")}'
                      for x in sw) or '-'
        print(f'{ep["id"]:26s} {ab:2s} {fmt(on["success"]):>3s} {fmt(ref and ref["success"]):>3s} '
              f'{on["actions"]:7.0f} {ref["actions"] if ref else float("nan"):6.0f}  {s}')
    print(f'\n{len(rows)} finished')
    for ab in 'AB':
        rs = [r for r in rows if ('A' if r[0]['goal_on_start_floor'] else 'B') == ab]
        if not rs:
            continue
        ok_on = [r[1]['success'] for r in rs if r[1]['success'] is not None]
        ok_ref = [r[2]['success'] for r in rs if r[2] and r[2]['success'] is not None]
        sw = [x for r in rs for x in r[3]]
        print(f'  {ab}: n {len(rs)}  SR on {100 * sum(ok_on) / max(len(ok_on), 1):.1f}% ({len(ok_on)} scored) '
              f'vs ref {100 * sum(ok_ref) / max(len(ok_ref), 1):.1f}%  | actions on {sum(r[1]["actions"] for r in rs) / len(rs):.0f} '
              f'vs ref {sum(r[2]["actions"] for r in rs if r[2]) / max(1, sum(1 for r in rs if r[2])):.0f}  | '
              f'episodes with a switch {sum(1 for r in rs if r[3])}, switches {len(sw)} '
              f'(right {sum(x["correct"] for x in sw)}, wrong {sum(not x["correct"] for x in sw)})')
    wrong_fail = [r[0]['id'] for r in rows if any(not x['correct'] for x in r[3]) and r[1]['success'] is False]
    if wrong_fail:
        print('  failed after a wrong switch: ' + ', '.join(wrong_fail))


if __name__ == '__main__':
    main()
