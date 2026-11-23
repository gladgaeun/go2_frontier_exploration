#!/usr/bin/env python3
# Probabilistic instance map (TravExplorer Sec. V-A) from open-vocabulary detections.
#
#   camera/detections (trav_interfaces/Detections: SAM3 instance masks + confidences, from
#   semantic_seg_node) + camera/depth/image_raw of the same stamp + camera_info + TF
#     -> each mask back-projected (depth <= max_range) into a point set P_q in the map frame
#     -> InstanceMap.update (eq. 4-6); a cluster of a query that lies in view (inside the image,
#        within max_range, not occluded in the depth image) but got no detection receives
#        negative evidence
#   Outputs (reliable clusters only, gamma >= min_conf):
#     instance_map/stair_landmarks  O_stair, landmark queries (default "stairs"): entrance x, y, z,
#                                   top x, y, z, points (the format of voxel_map/stair_landmarks)
#                                   + reliable (1 / 0: tentative, n >= landmark_min_count), belief
#                                   gamma n, gamma
#     instance_map/targets          O_tar, object queries: points x, y, z, id, gamma
#     instance_map/markers          every cluster (tentative ones grey) with label, gamma and n
#   Point clouds (every cluster, for RViz PointCloud2):
#     instance_map/points           P_y of every cluster (0.1 m voxel centres): x, y, z, rgb (label
#                                   colour, tentative grey), id, gamma
#     instance_map/detections       P_q of the last frame's detections (back-projected mask pixels,
#                                   not downsampled): x, y, z, rgb (label colour), score
#
# Detections come ~0.2-0.3 s after their depth frame (SAM3); depth frames are kept for 2 s.

import math
import threading

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from trav_exploration.instance_map import InstanceMap
from trav_exploration.recorder import Recorder
from trav_interfaces.msg import Detections


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _label_color(label, i):
    palette = [(0.95, 0.35, 0.2), (0.2, 0.6, 1.0), (0.95, 0.8, 0.1), (0.7, 0.3, 0.9), (0.2, 0.85, 0.5)]
    return palette[i % len(palette)] if label != 'stairs' else (0.0, 0.9, 0.9)


def _pack_rgb(r, g, b):
    """RViz 'rgb' field: 0x00RRGGBB as the bits of a float32."""
    v = np.array([(int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)], dtype=np.uint32)
    return float(v.view(np.float32)[0])


def _fields(names):
    return [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1) for k, n in enumerate(names)]


class InstanceMapNode(Node):

    def __init__(self):
        super().__init__('instance_map')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        # Q_land: landmark queries (-> O_stair); Q_obj: object queries (-> O_tar), comma-separated
        # (a string: launch passes the target, '' when there is none)
        self.landmark_queries = [q for q in p('landmark_queries', ['stairs']).value if q.strip()]
        self.object_queries = [q.split('=', 1)[0].strip() for q in str(p('object_queries', '').value).split(',')
                               if q.strip()]
        self.queries = tuple(self.landmark_queries + self.object_queries)
        self.max_range = float(p('max_range', 5.0).value)          # paper: depth 0-5 m
        self.stride = int(p('pixel_stride', 2).value)
        self.occlusion_tol = float(p('occlusion_tolerance', 0.15).value)
        self.erode = int(p('mask_erosion', 2).value)               # px: mask edges bleed onto the background
        self.map = InstanceMap(voxel=float(p('voxel', 0.1).value),
                               assoc_dist=float(p('assoc_distance', 1.0).value),
                               overlap=float(p('assoc_overlap', 0.2).value),
                               min_conf=float(p('min_confidence', 0.5).value),
                               min_count=float(p('min_count', 60).value),
                               min_visible=int(p('min_visible', 5).value),
                               min_detections=int(p('min_detections', 5).value),
                               score_weight=bool(p('score_weight', True).value),
                               reject_ratio=float(p('reject_ratio', 0.5).value))
        self.landmark_min_count = float(p('landmark_min_count', 100).value)
        # a tentative stair cluster is offered for verification (field reliable 0) only while its
        # fused confidence is at least landmark_tentative_min_conf: negative evidence (eq. 6,
        # stairs in view but not detected) then takes it out of O_stair, not only below min_conf
        # (00495: a 0.16 cluster was climbed for 600 s). The real stairs of 00810 hovered at
        # 0.49-0.50. Not in the paper (which keeps tentative hypotheses for later verification).
        self.landmark_tentative_min_conf = float(p('landmark_tentative_min_conf', 0.3).value)
        self.K = None
        self.depth = {}           # stamp (ns) -> (depth, frame id)
        self.dets = {}            # stamp (ns) -> Detections, waiting for their depth frame
        self.lock = threading.Lock()
        self.reliable_ids = set()
        self.n_frames = 0

        self.tf_buffer = Buffer()
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', parameter_overrides=[
            Parameter('use_sim_time', value=bool(self.get_parameter('use_sim_time').value))])
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        self.create_subscription(CameraInfo, p('camera_info_topic', 'camera/camera_info').value, self._on_info, 2)
        self.create_subscription(Image, p('depth_topic', 'camera/depth/image_raw').value, self._on_depth, 5)
        self.create_subscription(Detections, p('detections_topic', 'camera/detections').value, self._on_dets, 5)
        self.landmark_pub = self.create_publisher(PointCloud2, 'instance_map/stair_landmarks', 1)
        self.target_pub = self.create_publisher(PointCloud2, 'instance_map/targets', 1)
        self.marker_pub = self.create_publisher(MarkerArray, 'instance_map/markers', 1)
        self.points_pub = self.create_publisher(PointCloud2, 'instance_map/points', 1)
        self.det_cloud_pub = self.create_publisher(PointCloud2, 'instance_map/detections', 1)
        # record_dir: periodic state for offline datasets (recorder.py; '' = off)
        self.recorder = Recorder(self, 'instance_map', str(p('record_dir', '').value))
        if self.recorder:
            self.create_timer(float(p('record_period', 5.0).value), self._record_state)
        self.get_logger().info(f'instance map: landmarks {self.landmark_queries}, objects {self.object_queries}, '
                               f'min confidence {self.map.min_conf}')

    def _record_state(self):
        """Every instance (reliable or tentative) for offline datasets."""
        with self.lock:
            out = []
            for inst in self.map.instances:
                pts = self.map.points(inst.keys)
                if not len(pts):
                    continue
                out.append(dict(id=int(inst.id), label=inst.label, gamma=round(inst.gamma, 4), n=int(inst.n),
                                n_pos=int(inst.n_pos), n_neg=int(inst.n_neg), reliable=bool(self._usable(inst)),
                                centroid=pts.mean(axis=0).round(3), zmin=float(pts[:, 2].min()),
                                z10=float(np.percentile(pts[:, 2], 10)), zmax=float(pts[:, 2].max())))
        self.recorder.record('instances', instances=out)

    # -- inputs -----------------------------------------------------------------------------
    def _on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_depth(self, msg):
        if msg.encoding == '32FC1':
            d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, -1)[:, :msg.width]
        elif msg.encoding in ('16UC1', 'mono16'):
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, -1)[:, :msg.width] * 1e-3
        else:
            return
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        self.depth[key] = (d, msg.header.frame_id)
        while len(self.depth) > 40:                         # 2 s at 20 Hz
            del self.depth[min(self.depth)]
        if key in self.dets:
            self._process(self.dets.pop(key), *self.depth[key])

    def _on_dets(self, msg):
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        if key in self.depth:
            self._process(msg, *self.depth[key])
            return
        self.dets[key] = msg
        while len(self.dets) > 10:
            del self.dets[min(self.dets)]

    # -- one frame --------------------------------------------------------------------------
    def _process(self, msg, depth, frame):
        if self.K is None:
            return
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, frame, Time.from_msg(msg.header.stamp),
                                                timeout=Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warn(f'no TF for the detections: {e}', throttle_duration_sec=5.0)
            return
        tr = t.transform.translation
        cam_p, cam_R = np.array([tr.x, tr.y, tr.z]), quat_to_rot(t.transform.rotation)
        K = self.K
        h, w = msg.height, msg.width
        nb = (h * w + 7) // 8
        masks = np.frombuffer(bytes(msg.masks), dtype=np.uint8)
        dets = []
        from scipy import ndimage
        for i, (q, s) in enumerate(zip(msg.queries, msg.scores)):
            if q not in self.queries or len(masks) < (i + 1) * nb:
                continue
            m = np.unpackbits(masks[i * nb:(i + 1) * nb])[:h * w].reshape(h, w).astype(bool)
            if self.erode > 0:
                m = ndimage.binary_erosion(m, iterations=self.erode)
            m = m[::self.stride, ::self.stride]
            v, u = np.nonzero(m)
            v, u = v * self.stride, u * self.stride
            z = depth[v, u].astype(np.float64)
            ok = np.isfinite(z) & (z > 0.0)
            rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], axis=1)
            dist = z * np.linalg.norm(rays, axis=1)
            ok &= dist <= self.max_range
            if not ok.any():
                continue
            pts = (rays[ok] * z[ok, None]) @ cam_R.T + cam_p
            dets.append((q, pts, float(s)))

        def visible(pts):
            pc = (pts - cam_p) @ cam_R                   # map -> camera (optical) frame
            zc = pc[:, 2]
            ok = zc > 0.1
            u = np.full(len(pc), -1, dtype=np.int64)
            v = np.full(len(pc), -1, dtype=np.int64)
            u[ok] = np.round(K[0, 0] * pc[ok, 0] / zc[ok] + K[0, 2]).astype(np.int64)
            v[ok] = np.round(K[1, 1] * pc[ok, 1] / zc[ok] + K[1, 2]).astype(np.int64)
            ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h) & (np.linalg.norm(pc, axis=1) <= self.max_range)
            if not ok.any():
                return 0
            d = depth[v[ok], u[ok]].astype(np.float64)
            # seen: the depth image reaches the point (not occluded by something in front of it)
            seen = np.isfinite(d) & (d > 0.0) & (d >= zc[ok] - self.occlusion_tol)
            return int(seen.sum())

        self._publish_detections(Header(stamp=msg.header.stamp, frame_id=self.map_frame), dets)
        t_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self.lock:
            n_before = len(self.map.instances)
            self.map.update(dets, queries=self.queries, visible=visible, t=t_s)
            self.n_frames += 1
            for inst in self.map.instances[n_before:]:
                self.get_logger().info(f'[instance] new {inst.label} #{inst.id} at '
                                       f'({", ".join(f"{c:.2f}" for c in self.map.centroid(inst))}), '
                                       f'confidence {inst.gamma:.2f}')
            self._log_reliability()
        self._publish(Header(stamp=msg.header.stamp, frame_id=self.map_frame))

    def _usable(self, inst):
        if inst.n_pos < self.map.min_detections:            # one frame is no evidence yet
            return False
        if inst.label in self.landmark_queries:
            return inst.gamma >= self.map.min_conf and inst.n >= self.landmark_min_count
        return inst.gamma >= self.map.min_conf and inst.n >= self.map.min_count

    def _log_reliability(self):
        now = {i.id for i in self.map.instances if self._usable(i)}
        for inst in self.map.instances:
            if inst.id in now and inst.id not in self.reliable_ids:
                extra = ''
                if inst.label in self.landmark_queries:
                    lm = self.map.stair_landmark(inst)
                    extra = f', entrance ({lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f}), top z {lm[5]:.2f}'
                self.get_logger().info(f'[instance] {inst.label} #{inst.id} reliable: confidence {inst.gamma:.2f}, '
                                       f'n {inst.n:.0f} ({inst.n_pos} detections, {inst.n_neg} misses){extra}')
            elif inst.id in self.reliable_ids and inst.id not in now:
                self.get_logger().info(f'[instance] {inst.label} #{inst.id} back to tentative: confidence '
                                       f'{inst.gamma:.2f} ({inst.n_pos} detections, {inst.n_neg} misses)')
        self.reliable_ids = now

    # -- outputs ----------------------------------------------------------------------------
    def _colors(self):
        labels = sorted(set(self.queries))
        return {q: _label_color(q, labels.index(q)) for q in labels}

    def _publish_detections(self, h, dets):
        cols = self._colors()
        rows = []
        for q, pts, s in dets:
            rgb = _pack_rgb(*cols.get(q, (1.0, 1.0, 1.0)))
            rows.append(np.column_stack([pts.astype(np.float32), np.full(len(pts), rgb, dtype=np.float32),
                                         np.full(len(pts), s, dtype=np.float32)]))
        data = np.concatenate(rows) if rows else np.zeros((0, 5), dtype=np.float32)
        self.det_cloud_pub.publish(point_cloud2.create_cloud(h, _fields(('x', 'y', 'z', 'rgb', 'score')), data))

    def _publish(self, h):
        with self.lock:
            insts = list(self.map.instances)
            usable = [i for i in insts if self._usable(i)]
            # O_stair: reliable stair clusters, and the tentative ones with enough points (field
            # reliable 0) for later verification: the stairs of 00810 hovered at gamma 0.49-0.50
            # with 5700 points and were tentative when the floor was done, and the climb never came
            lms = [(*self.map.stair_landmark(i), 1.0 if self._usable(i) else 0.0, i.belief, i.gamma)
                   for i in sorted(insts, key=lambda i: -i.belief)
                   if i.label in self.landmark_queries and i.n >= self.landmark_min_count
                   and (self._usable(i) or i.gamma >= self.landmark_tentative_min_conf)]
            tgt = []
            for i in usable:
                if i.label in self.object_queries:
                    tgt += [(*c, float(i.id), i.gamma) for c in self.map.points(i.keys)]
            shapes = [(i, self.map.points(i.keys), self.map.centroid(i), self._usable(i)) for i in insts]
        f7 = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
              for k, n in enumerate(('x', 'y', 'z', 'top_x', 'top_y', 'top_z', 'size', 'reliable', 'belief',
                                     'gamma'))]
        self.landmark_pub.publish(point_cloud2.create_cloud(h, f7, [tuple(map(float, lm)) for lm in lms]))
        f5 = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
              for k, n in enumerate(('x', 'y', 'z', 'id', 'gamma'))]
        self.target_pub.publish(point_cloud2.create_cloud(h, f5, [tuple(map(float, t)) for t in tgt]))
        arr = MarkerArray()
        clear = Marker(header=h, action=Marker.DELETEALL)
        arr.markers.append(clear)
        labels = sorted({i.label for i, *_ in shapes})
        for inst, pts, c, ok in shapes:
            col = _label_color(inst.label, labels.index(inst.label)) if ok else (0.6, 0.6, 0.6)
            mk = Marker(header=h, ns='instances', id=2 * inst.id, type=Marker.CUBE_LIST, action=Marker.ADD)
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = self.map.voxel * 0.9
            mk.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.8 if ok else 0.35)
            mk.points = [Point(x=float(a), y=float(b), z=float(z)) for a, b, z in pts]
            arr.markers.append(mk)
            tx = Marker(header=h, ns='labels', id=2 * inst.id + 1, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            tx.pose.position.x, tx.pose.position.y = float(c[0]), float(c[1])
            tx.pose.position.z = float(pts[:, 2].max()) + 0.25
            tx.pose.orientation.w = 1.0
            tx.scale.z = 0.2
            tx.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0 if ok else 0.6)
            tx.text = f'{inst.label} {inst.gamma:.2f} (n {inst.n:.0f})' + ('' if ok else ' ?')
            arr.markers.append(tx)
        self.marker_pub.publish(arr)
        cols = self._colors()
        rows = []
        for inst, pts, c, ok in shapes:
            rgb = _pack_rgb(*(cols.get(inst.label, (1.0, 1.0, 1.0)) if ok else (0.6, 0.6, 0.6)))
            n = len(pts)
            rows.append(np.column_stack([pts.astype(np.float32), np.full(n, rgb, dtype=np.float32),
                                         np.full(n, inst.id, dtype=np.float32),
                                         np.full(n, inst.gamma, dtype=np.float32)]))
        data = np.concatenate(rows) if rows else np.zeros((0, 6), dtype=np.float32)
        self.points_pub.publish(point_cloud2.create_cloud(h, _fields(('x', 'y', 'z', 'rgb', 'id', 'gamma')), data))


def main(args=None):
    rclpy.init(args=args)
    node = InstanceMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
