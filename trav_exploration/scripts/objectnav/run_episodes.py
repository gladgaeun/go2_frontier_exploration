#!/usr/bin/env python3
# Run ObjectNav episodes one after another: for each episode of episodes.json
# (prepare_episodes.py) start habitat_server.py at the episode's start pose, launch the
# TravExplorer stack with the episode's target, wait until the robot stops at the target, the
# exploration ends, or a limit is hit, then shut everything down and keep the logs, the
# trajectory and result.json in <out>/<episode id>/. score_episodes.py computes SR / SPL.
#
# Run with the system Python (ROS 2 Humble), after colcon build, with the SAM3 and BLIP-2
# servers up (sam3 / blip2 conda envs):
#   python3 run_episodes.py --episodes ~/objectnav_runs/val20/episodes.json
#
# Finished episodes (result.json present) are skipped, so a run can be stopped and resumed.
# The run stops when a model server fails (SAM3 / BLIP-2 errors, "CUDA context lost"): every
# later episode would run blind.

import argparse
import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

HOME = os.path.expanduser('~')
WS = os.path.join(HOME, 'ros2_ws')
SERVER = os.path.join(WS, 'src/go2_frontier_exploration/habitat_bridge/scripts/habitat_server.py')

# launch output -> event
REACHED = re.compile(r'\[target\] reached (\S+) #(\d+): ([0-9.]+) m away')
EXPLORER_DONE = re.compile(r'\[explore_executor_node-\d+\]: process has finished cleanly')
MODEL_ERROR = re.compile(r'sam3_server error|blip2_server error')
NO_ANSWER = re.compile(r'no answer from (sam3|blip2)_server')
DIED = re.compile(r'\[ERROR\] \[(\S+?)-\d+\]: process has died')


def conda_cmd(env, cmd):
    return ['bash', '-c', f'source {HOME}/miniconda3/etc/profile.d/conda.sh 2>/dev/null || '
            f'source {HOME}/anaconda3/etc/profile.d/conda.sh; conda activate {env} && exec {cmd}']


def ros_cmd(cmd):
    return ['bash', '-c', f'source /opt/ros/humble/setup.bash && source {WS}/install/setup.bash && exec {cmd}']


def reader(proc, log_path, q):
    with open(log_path, 'w', buffering=1) as log:
        for line in iter(proc.stdout.readline, ''):
            log.write(line)
            q.put(line)
    q.put(None)


def stop(proc, name, timeout=20.0):
    if proc is None or proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            print(f'  {name} still running after {sig.name}')


def last_pose(traj):
    try:
        with open(traj) as f:
            lines = f.read().strip().splitlines()
        return [float(v) for v in lines[-1].split()] if lines else None
    except OSError:
        return None


class ActionCounter:
    """Actions of the discrete benchmark agent that the continuous motion amounts to (the paper:
    Habitat Challenge 2023, move_forward 0.25 m, turn 30 deg, at most 500 actions): distance
    travelled / 0.25 + rotation / 30 deg, from the trajectory written by habitat_server."""

    def __init__(self, path, step=0.25, turn=math.radians(30.0)):
        self.path, self.step, self.turn = path, step, turn
        self.pos = 0
        self.last = None
        self.dist = self.rot = 0.0
        self.t = 0.0

    def update(self):
        try:
            with open(self.path) as f:
                f.seek(self.pos)
                chunk = f.read()
        except OSError:
            return self.actions
        end = chunk.rfind('\n')
        if end < 0:
            return self.actions
        self.pos += len(chunk[:end + 1].encode())
        for line in chunk[:end].splitlines():
            v = [float(x) for x in line.split()]
            if self.last is not None:
                self.dist += math.hypot(v[1] - self.last[1], v[3] - self.last[3])
                self.rot += abs((v[4] - self.last[4] + math.pi) % (2.0 * math.pi) - math.pi)
            self.last, self.t = v, v[0]
        return self.actions

    @property
    def actions(self):
        return self.dist / self.step + self.rot / self.turn


def run_episode(ep, args, d):
    os.makedirs(d, exist_ok=True)
    traj = os.path.join(d, 'trajectory.txt')
    start = ','.join(f'{v:.4f}' for v in ep['start']) + f",{ep['yaw']:.4f}"
    # the agent moves on the scene's own navmesh, the one the episodes (start, geodesic distance)
    # and the scoring use: a navmesh recomputed for radius 0.18 cut the narrow top of the stairs of
    # 00803 off its upper floor (goal 1.2 m away on the dataset navmesh, 66 m on the recomputed one)
    server = subprocess.Popen(
        conda_cmd(args.habitat_env, f'python -u {SERVER} --scene {ep["glb"]} --no-recompute-navmesh '
                                    f'--start={start} --bind {args.habitat_address} --trajectory {traj}'),
        stdout=open(os.path.join(d, 'server.log'), 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        with open(os.path.join(d, 'server.log')) as f:
            if 'listening on' in f.read():
                break
        if server.poll() is not None:
            break
        time.sleep(1.0)
    else:
        stop(server, 'habitat_server')
        return {'status': 'error', 'reason': 'habitat_server did not start'}
    if server.poll() is not None:
        return {'status': 'error', 'reason': 'habitat_server exited at start'}

    target = shlex.quote(f"target:={ep['category']}={ep['prompt']}")      # 'potted plant' has a space
    launch_args = (f"rviz:={'true' if args.rviz else 'false'} segmentation:=sam3 {target} "
                   f"zmq_address:={args.habitat_address} sam3_address:={args.sam3_address} "
                   f"blip2_address:={args.blip2_address} explore_restarts:={args.explore_restarts} "
                   f"map_z_below:={ep['map_z_below']} map_z_above:={ep['map_z_above']} "
                   + (f"record_dir:={os.path.join(d, 'record')} " if args.record else '') + args.launch_args)
    env = dict(os.environ, RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', ROS_DOMAIN_ID=str(args.domain_id),
               PYTHONUNBUFFERED='1')
    launch = subprocess.Popen(ros_cmd(f'ros2 launch trav_exploration hm3d_trav_exploration.launch.py {launch_args}'),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                              start_new_session=True)
    q = queue.Queue()
    threading.Thread(target=reader, args=(launch, os.path.join(d, 'launch.log'), q), daemon=True).start()

    # the launch runs in a session of its own (start_new_session): a Ctrl+C of this script does
    # not reach it, so it is stopped here in every case (an interrupted run left bridge_node
    # stepping the next episode's server at twice the rate)
    try:
        res = {'status': None, 'stop_called': False}
        counter = ActionCounter(traj)
        n_explorer_done, n_no_answer, t_start = 0, 0, time.time()
        while res['status'] is None:
            try:
                line = q.get(timeout=1.0)
            except queue.Empty:
                line = ''
            if line is None:
                res.update(status='error', reason='launch exited')
                break
            m = REACHED.search(line)
            if m:
                res.update(status='stopped', stop_called=True, reached_label=m.group(1),
                           reached_instance=int(m.group(2)), reached_distance=float(m.group(3)))
                break
            if EXPLORER_DONE.search(line):
                n_explorer_done += 1
                if n_explorer_done > args.explore_restarts:
                    res.update(status='exploration_ended')
            if MODEL_ERROR.search(line):
                res.update(status='model_error', reason=line.strip()[-300:])
            if NO_ANSWER.search(line):
                n_no_answer += 1
                if n_no_answer >= 5:                       # one is a busy server, five a dead one
                    res.update(status='model_error', reason=line.strip()[-300:])
            m = DIED.search(line)
            # RViz is a viewer: it segfaults now and then (GLSL link error of the map display)
            if m and 'explore_executor_node' not in m.group(1) and 'rviz2' not in m.group(1):
                res.update(status='error', reason=line.strip()[-300:])
            if counter.update() >= args.max_actions:
                res.update(status='action_limit')
            if counter.t > args.max_sim_time:
                res.update(status='sim_time_limit')
            if time.time() - t_start > args.wall_timeout:
                res.update(status='wall_timeout')
        time.sleep(1.0)                                    # the last poses reach the trajectory
    finally:
        stop(launch, 'ros2 launch')
        stop(server, 'habitat_server')
    pose = last_pose(traj)
    counter.update()
    res.update(wall_time=round(time.time() - t_start, 1), sim_time=pose[0] if pose else None,
               final_pose=pose[1:] if pose else None, actions=round(counter.actions, 1))
    return res


def leftovers(domain_id):
    """Processes of an earlier run on this ROS domain (a bridge_node left over steps the next
    episode's habitat server too, and its /clock and TF mix with the new ones)."""
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode(errors='ignore')
            if not re.search(r'install/(trav_exploration|frontier_exploration|habitat_bridge)/lib|'
                             r'hm3d_trav_exploration\.launch', cmd):
                continue
            with open(f'/proc/{pid}/environ', 'rb') as f:
                env = dict(kv.split(b'=', 1) for kv in f.read().split(b'\0') if b'=' in kv)
        except OSError:
            continue
        if env.get(b'ROS_DOMAIN_ID', b'0').decode() == str(domain_id):
            out.append(f'{pid} {cmd[:120]}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', required=True, help='episodes.json from prepare_episodes.py')
    p.add_argument('--out', default='', help='results directory (default: next to episodes.json)')
    p.add_argument('--only', default='', help='comma-separated episode ids to run')
    p.add_argument('--max-actions', type=float, default=500.0,
                   help='episode limit in benchmark actions (distance / 0.25 m + rotation / 30 deg; the '
                        'paper / Habitat Challenge 2023: 500)')
    p.add_argument('--max-sim-time', type=float, default=1200.0, help='safety limit, s of simulated time')
    p.add_argument('--wall-timeout', type=float, default=2400.0)
    p.add_argument('--explore-restarts', type=int, default=2)
    p.add_argument('--habitat-env', default='hsim-test')
    p.add_argument('--habitat-address', default='tcp://127.0.0.1:5575')
    p.add_argument('--sam3-address', default='tcp://127.0.0.1:5557')
    p.add_argument('--blip2-address', default='tcp://127.0.0.1:5558')
    p.add_argument('--domain-id', type=int, default=57, help='ROS_DOMAIN_ID (keeps other ROS sessions apart)')
    p.add_argument('--rviz', action='store_true')
    p.add_argument('--launch-args', default='', help='extra launch arguments')
    p.add_argument('--record', action='store_true',
                   help='state records of voxel_mapper / frontier_goal / instance_map in <episode>/record '
                        '(offline datasets, recorder.py)')
    args = p.parse_args()

    eps = json.load(open(os.path.expanduser(args.episodes)))
    out = os.path.expanduser(args.out) or os.path.dirname(os.path.abspath(os.path.expanduser(args.episodes)))
    only = {s for s in args.only.split(',') if s}
    todo = [e for e in eps if not only or e['id'] in only]
    left = leftovers(args.domain_id)
    if left:
        print(f'processes of an earlier run on ROS_DOMAIN_ID {args.domain_id}; stop them first:\n  '
              + '\n  '.join(left))
        sys.exit(3)
    print(f'{len(todo)} episodes -> {out}')
    for k, ep in enumerate(todo):
        d = os.path.join(out, ep['id'])
        rp = os.path.join(d, 'result.json')
        if os.path.exists(rp):
            continue
        print(f"[{k + 1}/{len(todo)}] {ep['id']}: {ep['category']} ({'cross-floor' if ep['cross_floor'] else 'same floor'}, "
              f"geodesic {ep['geodesic_distance']:.1f} m)", flush=True)
        try:
            res = run_episode(ep, args, d)
        except KeyboardInterrupt:
            print('interrupted; this episode is not saved')
            subprocess.run(['pkill', '-INT', '-f', f'--bind {args.habitat_address}'])
            sys.exit(1)
        res.update(id=ep['id'], category=ep['category'])
        print(f"  -> {res['status']}" + (f" ({res.get('reason', '')})" if res.get('reason') else '')
              + f", sim {res.get('sim_time')} s, wall {res.get('wall_time')} s", flush=True)
        if res['status'] == 'model_error':
            print('a model server failed: stopping the run (restart the server, then run again to resume)')
            sys.exit(2)
        with open(rp, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
