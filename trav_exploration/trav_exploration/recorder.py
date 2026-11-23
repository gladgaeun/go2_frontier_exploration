"""JSON Lines state recorder for offline datasets (floor-switch model data collection).

Each node that records gets a `record_dir` parameter ('' = off) and writes
<record_dir>/<name>.jsonl: a header line (node, git commit of the source tree, wall time), then
one line per record() call with the ROS time (sim time with use_sim_time) and the given fields.
numpy scalars / arrays are written as numbers / lists. Decisions are in the launch log; the
records are periodic snapshots of the state those decisions were made on."""
import json
import os
import subprocess
import time

import numpy as np


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def source_commit():
    """(commit, dirty) of the git tree this module is in (symlink install: the source tree)."""
    here = os.path.dirname(os.path.realpath(__file__))
    try:
        c = subprocess.run(['git', '-C', here, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                           timeout=5).stdout.strip()
        d = subprocess.run(['git', '-C', here, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return c or None, bool(d)
    except (OSError, subprocess.SubprocessError):
        return None, None


class Recorder:
    def __init__(self, node, name, record_dir):
        self.node = node
        self.f = None
        if not record_dir:
            return
        os.makedirs(record_dir, exist_ok=True)
        self.f = open(os.path.join(record_dir, f'{name}.jsonl'), 'w', buffering=1)
        commit, dirty = source_commit()
        self.f.write(json.dumps({'type': 'header', 'node': name, 'commit': commit, 'dirty': dirty,
                                 'wall': time.time()}) + '\n')

    def __bool__(self):
        return self.f is not None

    def record(self, kind, **fields):
        if self.f is None:
            return
        t = self.node.get_clock().now().nanoseconds * 1e-9
        fields = _plain(fields)
        fields.update(type=kind, t=round(t, 3))
        try:
            self.f.write(json.dumps(fields, allow_nan=False) + '\n')
        except (ValueError, OSError) as e:
            self.node.get_logger().warn(f'[record] {kind} not written: {e}', throttle_duration_sec=30.0)
