#!/usr/bin/env python3
"""Online floor-switch node (v0): p, the probability that the target is on the robot's floor, and
whether the switch rule is ready. frontier_goal_node decides at its decision times.

Inputs, the same as the offline features:
  record/frontier_goal, record/instance_map, record/voxel_mapper (std_msgs/String): the JSON
    records of trav_exploration's recorder (record_publish:=true on those nodes). Each
    frontier_goal 'state' record (1 s) is one decision time t, handled `delay` s (wall) after it
    came in; the latest instance and voxel map records with a time <= t are taken with it, as
    scripts/make_features.py does (not a newer one that arrived before the handling).
  TF map -> base_link every 0.2 s: benchmark actions so far (distance / 0.25 m + rotation / 30 deg,
    as run_episodes counts them from the trajectory).
  camera/image_raw + camera/depth/image_raw (same stamp) and TF map -> camera optical frame: the
    room-type frame of the decision time: the latest pair stamped <= t (render_frames.py: the
    pose at the latest trajectory sample <= t). blip2_server's "itc" request gives the raw similarity to
    each room prompt; the depth (every 4th pixel, as render_frames.py) puts it on the floor's room
    map (rooms.RoomMap, margin and T of online_model.json).
Per decision time: features.FloorFeatures (15) + the room histogram (9) -> model.PModel (MLP32 +
Platt) -> rule.SwitchRule (last X p below tau, cost guard). Published on floor_switch/state
(std_msgs/String, JSON): t, p, logit, floor_z, ready, n_low (history length), stairs (m), need,
actions. floor_switch/markers (visualization_msgs/MarkerArray, RViz): above the robot, the p, the
count of low p of the history (n_low / X), ready and the deferred floors, and a sphere coloured by
the state (green p >= tau, orange p below tau, red ready, grey no estimate). With record_dir: <record_dir>/floor_switch.jsonl, the same with the 24 inputs (online
vs offline check).
"""
import json
import math
import queue
import threading
import time
from collections import deque

import numpy as np
import rclpy
import zmq
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformException, TransformListener

from floor_switch.features import FloorFeatures
from floor_switch.model import PModel
from floor_switch.rooms import PROMPTS, RoomMap, optical_points
from floor_switch.rule import SwitchRule, stair_distance

try:
    from trav_exploration.recorder import Recorder
except ImportError:                       # floor_switch without trav_exploration: no record file
    Recorder = None


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class FloorSwitchNode(Node):

    def __init__(self):
        super().__init__('floor_switch')
        p = self.declare_parameter
        self.model = PModel(str(p('model', '').value))
        r = self.model.meta['rule']
        self.rule = SwitchRule(r['X'], r['tau'], r['actions_per_metre'], r['climb_p75'] + r['margin'],
                               float(p('max_actions', 500.0).value))
        rm = self.model.meta['rooms']
        self.T = float(rm['T'])
        self.rooms = RoomMap([self.T], res=float(rm['res']), margin_min=float(rm['margin']))
        self.map_frame = str(p('map_frame', 'map').value)
        self.base_frame = str(p('base_frame', 'base_link').value)
        self.stride = int(p('depth_stride', 4).value)
        self.address = str(p('blip2_address', 'tcp://127.0.0.1:5558').value)
        self.timeout_ms = int(1000 * float(p('timeout', 3.0).value))
        self.feats = None                  # FloorFeatures, made at the first state (its target)
        self.delay = float(p('delay', 0.3).value)
        self.inst, self.vm = deque(maxlen=20), deque(maxlen=20)      # (t, record), by arrival
        self.actions, self.last_pose = 0.0, None
        self.rgb, self.depth = {}, {}      # stamp (ns) -> message, last few
        self.hfov = None
        self.lock = threading.Lock()
        self.jobs = queue.Queue()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(String, 'record/frontier_goal', self._on_fg, 50)
        self.create_subscription(String, 'record/instance_map', self._on_inst, 10)
        self.create_subscription(String, 'record/voxel_mapper', self._on_vm, 10)
        self.create_subscription(Image, p('image_topic', 'camera/image_raw').value, self._on_rgb, 5)
        self.create_subscription(Image, p('depth_topic', 'camera/depth/image_raw').value, self._on_depth, 5)
        self.create_subscription(CameraInfo, p('camera_info_topic', 'camera/camera_info').value, self._on_info, 1)
        self.pub = self.create_publisher(String, 'floor_switch/state', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'floor_switch/markers', 2)
        self.create_timer(0.2, self._count_actions)
        self.recorder = None
        if Recorder is not None:
            self.recorder = Recorder(self, 'floor_switch', str(p('record_dir', '').value))
        self.ctx = zmq.Context()
        self.sock = None
        self._connect()
        threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info(
            f'floor switch v0: {self.model.meta["model"]}; switch ready when the last {self.rule.X} p '
            f'< {self.rule.tau:.2f} and actions left >= {self.rule.k:.2f} x stairs (m) + {self.rule.fixed:.0f}; '
            f'rooms: blip2_server at {self.address}, margin {rm["margin"]}, T {self.T}')

    # -- inputs ---------------------------------------------------------------------------------
    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

    def _on_fg(self, msg):
        r = json.loads(msg.data)
        if r.get('type') == 'state':
            self.jobs.put((time.time(), r))

    def _on_inst(self, msg):
        r = json.loads(msg.data)
        if r.get('type') == 'instances':
            with self.lock:
                self.inst.append((r['t'], r.get('instances') or []))

    def _on_vm(self, msg):
        r = json.loads(msg.data)
        if r.get('type') == 'state':
            with self.lock:
                self.vm.append((r['t'], r))

    def _on_info(self, msg):
        if msg.k[0] > 0.0:
            self.hfov = math.degrees(2.0 * math.atan(0.5 * msg.width / msg.k[0]))

    @staticmethod
    def _key(msg):
        return msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec

    def _keep(self, cache, msg):
        with self.lock:
            cache[self._key(msg)] = msg
            while len(cache) > 40:                  # ~2 s of frames
                del cache[min(cache)]

    def _on_rgb(self, msg):
        self._keep(self.rgb, msg)

    def _on_depth(self, msg):
        self._keep(self.depth, msg)

    def _count_actions(self):
        """Benchmark actions from the robot's motion in the map frame (run_episodes.ActionCounter)."""
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        x, y = t.transform.translation.x, t.transform.translation.y
        if self.last_pose is not None:
            d = math.hypot(x - self.last_pose[0], y - self.last_pose[1])
            r = abs((yaw - self.last_pose[2] + math.pi) % (2.0 * math.pi) - math.pi)
            self.actions += d / 0.25 + r / math.radians(30.0)
        self.last_pose = (x, y, yaw)

    # -- room frame -----------------------------------------------------------------------------
    def _frame(self, t):
        """(sims (9,), points (N, 3), confidences (N,)) of the latest RGB-D pair stamped <= t, or None."""
        with self.lock:
            keys = [k for k in sorted(set(self.rgb) & set(self.depth)) if k <= round(t * 1e9) + 10 ** 6]
            if not keys or self.hfov is None:
                return None
            rgb, dep = self.rgb[keys[-1]], self.depth[keys[-1]]
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, dep.header.frame_id, Time.from_msg(dep.header.stamp),
                                                timeout=Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(f'no TF for the room frame: {e}', throttle_duration_sec=10.0)
            return None
        if dep.encoding == '32FC1':
            d = np.frombuffer(dep.data, dtype=np.float32).reshape(dep.height, -1)[:, :dep.width]
        elif dep.encoding in ('16UC1', 'mono16'):
            d = np.frombuffer(dep.data, dtype=np.uint16).reshape(dep.height, -1)[:, :dep.width] * 1e-3
        else:
            return None
        st = self.stride
        d = d[st // 2::st, st // 2::st]
        tr = t.transform.translation
        pts, conf = optical_points(d, self.hfov, (tr.x, tr.y, tr.z), quat_to_R(t.transform.rotation))
        if rgb.encoding != 'rgb8':
            self.get_logger().error(f'expected rgb8, got {rgb.encoding}', throttle_duration_sec=10.0)
            return None
        img = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, -1)[:, :3 * rgb.width]
        self.sock.send_multipart([json.dumps({'op': 'itc', 'width': rgb.width, 'height': rgb.height,
                                              'texts': list(PROMPTS)}).encode(), np.ascontiguousarray(img).tobytes()])
        if not self.sock.poll(self.timeout_ms):
            self.get_logger().warn(f'no answer from blip2_server at {self.address} (itc); retrying',
                                   throttle_duration_sec=5.0)
            self._connect()
            return None
        hdr = json.loads(self.sock.recv_multipart()[0].decode())
        if 'error' in hdr:
            self.get_logger().error(f"blip2_server error: {hdr['error']}", throttle_duration_sec=5.0)
            return None
        return np.asarray(hdr['scores'], dtype=np.float64), pts, conf

    # -- decision times -------------------------------------------------------------------------
    def _worker(self):
        while rclpy.ok():
            try:
                wall, fg = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            while not self.jobs.empty():           # behind: the newest state only (logged)
                wall, fg = self.jobs.get()
                self.get_logger().warn('decision times skipped (behind)', throttle_duration_sec=10.0)
            wait = wall + self.delay - time.time()   # the records and frames of time t come in
            if wait > 0.0:
                time.sleep(wait)
            try:
                self._step(fg)
            except Exception as e:                 # noqa: BLE001 (keep running; logged)
                self.get_logger().error(f'step failed: {e!r}', throttle_duration_sec=5.0)

    def _step(self, fg):
        target = fg.get('target') or ''
        if self.feats is None:
            if not target:
                return
            self.feats = FloorFeatures(target)
        t = float(fg['t'])
        with self.lock:
            inst = max((x for x in self.inst if x[0] <= t + 1e-6), key=lambda x: x[0], default=(None, []))[1]
            vm = max((x for x in self.vm if x[0] <= t + 1e-6), key=lambda x: x[0], default=(None, {}))[1]
        actions = self.actions
        x = self.feats.step(fg, inst, vm, actions)
        z = fg.get('z_r')
        climbing = bool(vm.get('climbing'))
        hist, sims = None, None
        if z is not None:
            fr = self._frame(t)
            if fr is None:
                self.rooms.update(np.zeros(len(PROMPTS)), np.zeros((0, 3)), np.zeros(0), z, climbing)
            else:
                sims = fr[0]
                self.rooms.update(fr[0], fr[1], fr[2], z, climbing)
            hist = self.rooms.histogram(z, self.T)
        p = logit = None
        floor = None
        if x is not None:
            p, logit = self.model.p(np.concatenate([x, hist]), self.feats.target)
            floor = self.feats.ft.find(z)
        stairs = stair_distance(fg)
        tracking = bool(fg.get('tracking_target'))
        ready, need = self.rule.step(p, floor, tracking, stairs, actions)
        if fg.get('target_found'):
            ready = False
        out = dict(t=fg.get('t'), p=p, logit=logit, floor_z=None if floor is None else self.feats.floors[floor],
                   z_r=z, ready=bool(ready), n_low=sum(1 for v in self.rule.hist if v < self.rule.tau),
                   n_hist=len(self.rule.hist), stairs=stairs, need=need, actions=round(actions, 1),
                   tracking=tracking)
        self.pub.publish(String(data=json.dumps(out)))
        self._markers(fg, out)
        if self.recorder:
            self.recorder.record('step', **out, fg_t=fg.get('t'),
                                 x=None if x is None else [round(float(v), 5) for v in x],
                                 rooms=None if hist is None else [round(float(v), 5) for v in hist],
                                 sims=None if sims is None else [round(float(v), 5) for v in sims])


    def _markers(self, fg, out):
        rob = fg.get('robot') or [None, None]
        if rob[0] is None or out['z_r'] is None:
            return
        p, X, tau = out['p'], self.rule.X, self.rule.tau
        if p is None:
            col, state = (0.6, 0.6, 0.6), 'no estimate' if not out['tracking'] else 'tracking target'
        elif out['ready']:
            col, state = (1.0, 0.1, 0.1), 'SWITCH READY'
        elif p < tau:
            col, state = (1.0, 0.6, 0.0), f'low {out["n_low"]}/{X}'
        else:
            col, state = (0.1, 0.8, 0.2), f'stay (low {out["n_low"]}/{X})'
        if out['tracking']:
            state = 'tracking target'
        lines = [f'p {p:.2f} (tau {tau:.2f})' if p is not None else 'p -', state]
        if out['ready'] or out['need'] is not None:
            lines.append(f'stairs {out["stairs"]:.1f} m, actions {out["actions"]:.0f}'
                         + (f' (left needed {out["need"]:.0f})' if out['need'] is not None else ''))
        dfl = fg.get('deferred_floors') or []
        if dfl:
            lines.append('deferred floor z ' + ', '.join(f'{z:.1f}' for z in dfl))
        x, y, z = float(rob[0]), float(rob[1]), float(out['z_r'])
        ma = MarkerArray()
        txt = Marker()
        txt.header.frame_id = self.map_frame
        txt.ns, txt.id, txt.type, txt.action = 'floor_switch', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, z + 1.6
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.25
        txt.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
        txt.text = '\n'.join(lines)
        sph = Marker()
        sph.header.frame_id = self.map_frame
        sph.ns, sph.id, sph.type, sph.action = 'floor_switch', 1, Marker.SPHERE, Marker.ADD
        sph.pose.position.x, sph.pose.position.y, sph.pose.position.z = x, y, z + 1.15
        sph.pose.orientation.w = 1.0
        r = 0.15 + 0.25 * (1.0 - (p if p is not None else 1.0))     # bigger as p drops
        sph.scale.x = sph.scale.y = sph.scale.z = r
        sph.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.9)
        ma.markers = [txt, sph]
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = FloorSwitchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
