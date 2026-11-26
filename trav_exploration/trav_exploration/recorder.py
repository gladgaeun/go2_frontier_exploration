"""JSON Lines state recorder for offline datasets (floor-switch model data collection).

Each node that records gets a `record_dir` parameter ('' = off) and writes
<record_dir>/<name>.jsonl: a header line (node, git commit of the source tree, wall time), then
one line per record() call with the ROS time (sim time with use_sim_time), the wall time and the
given fields.
With the node parameter record_publish (default false) every record is also published, as the
same JSON line, on record/<name> (std_msgs/String): the online floor-switch node
(floor_switch/floor_switch_node.py) reads the state the offline features were made from. The
recorder is on when either output is.
numpy scalars / arrays are written as numbers / lists. Decisions are in the launch log; the
records are periodic snapshots of the state those decisions were made on."""
import json
import os
import subprocess
import time

import numpy as np
from std_msgs.msg import String


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
        self.pub = None
        if node.declare_parameter('record_publish', False).value:
            self.pub = node.create_publisher(String, f'record/{name}', 10)
        if not record_dir:
            return
        os.makedirs(record_dir, exist_ok=True)
        # append: a node restarted within the episode (the exploration loop) keeps its records
        self.f = open(os.path.join(record_dir, f'{name}.jsonl'), 'a', buffering=1)
        commit, dirty = source_commit()
        self.f.write(json.dumps({'type': 'header', 'node': name, 'commit': commit, 'dirty': dirty,
                                 'wall': time.time()}) + '\n')

    def __bool__(self):
        return self.f is not None or self.pub is not None

    def record(self, kind, **fields):
        if not self:
            return
        t = self.node.get_clock().now().nanoseconds * 1e-9
        fields = _plain(fields)
        fields.update(type=kind, t=round(t, 3), wall=round(time.time(), 3))
        try:
            line = json.dumps(fields, allow_nan=False)
            if self.f is not None:
                self.f.write(line + '\n')
            if self.pub is not None:
                self.pub.publish(String(data=line))
        except (ValueError, OSError) as e:
            self.node.get_logger().warn(f'[record] {kind} not written: {e}', throttle_duration_sec=30.0)
