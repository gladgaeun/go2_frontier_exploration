#!/usr/bin/env python3
# ROS 2 side of the Habitat <-> ROS 2 bridge. Runs with the system Python (ROS 2 Humble).
#
# Every timer tick it sends the latest /cmd_vel to habitat_server.py over ZMQ, then publishes
# what came back: /clock, /odom, TF odom->base_link, /scan and (optionally) /camera/image_raw.
# With mapper:=gt (default) it also builds /map from the exact pose and publishes a static
# identity map->odom, so no SLAM node is needed in simulation (see gt_mapper.py).
# This node itself runs on wall time (use_sim_time:=false); everything else uses /clock.

import json
import math

import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import PolygonStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from habitat_bridge.gt_mapper import GridMapper


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class HabitatBridge(Node):

    def __init__(self):
        super().__init__('habitat_bridge')
        self.address = self.declare_parameter('zmq_address', 'tcp://127.0.0.1:5555').value
        self.rate = float(self.declare_parameter('rate_hz', 20.0).value)
        self.cmd_timeout = float(self.declare_parameter('cmd_timeout', 0.5).value)
        self.recv_timeout_ms = int(1000 * float(self.declare_parameter('recv_timeout', 5.0).value))
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.laser_frame = self.declare_parameter('laser_frame', 'laser').value
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_link').value
        self.publish_rgb = bool(self.declare_parameter('publish_rgb', True).value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        # mapper 'gt': this node publishes /map and a static identity map->odom (exact odometry).
        # mapper 'slam_toolbox': slam_toolbox owns both; this node publishes neither. Never both:
        # two publishers of map->odom made Nav2 lose map->base_link and the robot froze/spun.
        self.mapper_mode = str(self.declare_parameter('mapper', 'gt').value)
        self.publish_map_to_odom = self.mapper_mode == 'gt'
        self.map_period = float(self.declare_parameter('map_publish_period', 1.0).value)
        self.map_integrate_every = int(self.declare_parameter('map_integrate_every', 2).value)
        self.mapper = GridMapper(resolution=float(self.declare_parameter('map_resolution', 0.05).value)) \
            if self.mapper_mode == 'gt' else None
        self.tick_count = 0
        self.last_map_pub = -1e9

        if self.get_parameter('use_sim_time').value:
            self.get_logger().warn('habitat_bridge must run with use_sim_time:=false (it publishes /clock)')

        self.dt = 1.0 / self.rate
        self.cmd = (0.0, 0.0)
        self.cmd_sim_time = -1e9
        self.sim_time = 0.0
        self.info = None

        self.ctx = zmq.Context()
        self.sock = None
        self._connect()

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.img_pub = self.create_publisher(Image, 'camera/image_raw', 2)
        self.tf_pub = TransformBroadcaster(self)
        self.static_tf_pub = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10)
        # Watchdog: the local costmap footprint is stamped with the controller's latest TF.
        # If that lags the simulation clock, Nav2 is driving on a stale pose.
        self.footprint_stamp = None
        self.last_lag_warn = -1e9
        self.create_subscription(PolygonStamped, 'local_costmap/published_footprint',
                                 self._on_footprint, 10)
        # Loop-back check: does our own odom->base_link come back over DDS? If yes while the
        # controller is stale, the problem is on the receiving side, not in this bridge.
        self.own_tf_stamp = None
        self.create_subscription(TFMessage, '/tf', self._on_own_tf, 100)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', map_qos) if self.mapper else None

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(f'Habitat bridge -> {self.address} at {self.rate:.0f} Hz')

    # -- ZMQ -------------------------------------------------------------------
    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

    def _request(self, msg):
        """Send one request, return (header, scan_bytes, rgb_bytes) or None on timeout."""
        self.sock.send(json.dumps(msg).encode())
        if not self.sock.poll(self.recv_timeout_ms):
            self.get_logger().warn(
                f'No answer from habitat_server at {self.address} - is it running? retrying...',
                throttle_duration_sec=5.0)
            self._connect()  # a REQ socket cannot send again until it got a reply
            return None
        parts = self.sock.recv_multipart()
        hdr = json.loads(parts[0].decode())
        if 'error' in hdr:
            self.get_logger().error(f"habitat_server error: {hdr['error']}")
            return None
        return hdr, parts[1], parts[2]

    # -- callbacks -------------------------------------------------------------
    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.cmd_sim_time = self.sim_time

    def _on_footprint(self, msg):
        self.footprint_stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def _on_own_tf(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == self.base_frame:
                self.own_tf_stamp = t.header.stamp.sec + 1e-9 * t.header.stamp.nanosec

    def _check_controller_tf(self):
        if self.footprint_stamp is None:
            return
        lag = self.sim_time - self.footprint_stamp
        if lag > 2.0 and self.sim_time - self.last_lag_warn > 5.0:
            self.last_lag_warn = self.sim_time
            self.get_logger().error(
                f'[watchdog] controller TF is {lag:.1f}s behind the simulation '
                f'(frozen at t={self.footprint_stamp:.2f}s) - Nav2 is driving on a stale pose. '
                f'/tf loop-back here: t={self.own_tf_stamp if self.own_tf_stamp is not None else -1:.2f}s '
                f'(sim t={self.sim_time:.2f}s)')

    def _stamp(self):
        return Time(nanoseconds=int(round(self.sim_time * 1e9))).to_msg()

    def _tick(self):
        if self.info is None:
            res = self._request({'op': 'info'})
            if res is None:
                return
            self.info = res[0]
            self.get_logger().info(f'Connected to habitat_server: {self.info}')
            self._publish_static_tf()
            res = self._request({'op': 'reset'})
        else:
            vx, wz = self.cmd
            if self.sim_time - self.cmd_sim_time > self.cmd_timeout:
                vx = wz = 0.0  # stop if Nav2 stopped talking
            res = self._request({'op': 'step', 'vx': vx, 'wz': wz, 'dt': self.dt})
        if res is None:
            return

        hdr, scan_bytes, rgb_bytes = res
        self.sim_time = hdr['sim_time']
        stamp = self._stamp()
        self.clock_pub.publish(Clock(clock=stamp))
        self._publish_odom(hdr['odom'], stamp)
        ranges = np.frombuffer(scan_bytes, dtype=np.float32)
        self._check_controller_tf()
        self._publish_scan(ranges, stamp)
        if self.mapper is not None:
            self._update_map(hdr['odom'], ranges, stamp)
        if self.publish_rgb and rgb_bytes:
            self._publish_rgb(rgb_bytes, stamp)

    # -- mapping ---------------------------------------------------------------
    def _update_map(self, o, ranges, stamp):
        self.tick_count += 1
        n = len(ranges)
        if n and self.tick_count % self.map_integrate_every == 0:
            # laser frame = base_link shifted up only, odom frame = map frame
            self.mapper.integrate(o['x'], o['y'], o['yaw'], ranges, -math.pi,
                                  2.0 * math.pi / n, float(self.info['max_range']))
        if self.sim_time - self.last_map_pub >= self.map_period:
            g = self.mapper.grid()
            if g is None:
                return
            data, w, h, ox, oy = g
            m = OccupancyGrid()
            m.header.stamp = stamp
            m.header.frame_id = self.map_frame
            m.info.map_load_time = stamp
            m.info.resolution = self.mapper.res
            m.info.width, m.info.height = int(w), int(h)
            m.info.origin.position.x = float(ox)
            m.info.origin.position.y = float(oy)
            m.info.origin.orientation.w = 1.0
            m.data = data.tolist()
            self.map_pub.publish(m)
            self.last_map_pub = self.sim_time
            self.map_count = getattr(self, 'map_count', 0) + 1
            if self.map_count % 10 == 0:   # exploration progress, compare with the server's navigable area
                free_m2 = float((data == 0).sum()) * self.mapper.res ** 2
                self.get_logger().info(f'[explore] t_sim={self.sim_time:6.1f}s  mapped free area {free_m2:6.1f} m^2')

    # -- publishers ------------------------------------------------------------
    def _publish_static_tf(self):
        tfs = []
        if self.publish_map_to_odom:
            t = TransformStamped()
            t.header.stamp = self._stamp()
            t.header.frame_id = self.map_frame
            t.child_frame_id = self.odom_frame
            t.transform.rotation.w = 1.0
            tfs.append(t)
        for child, z in ((self.laser_frame, self.info['laser_height']),
                         (self.camera_frame, self.info['camera_height'])):
            t = TransformStamped()
            t.header.stamp = self._stamp()
            t.header.frame_id = self.base_frame
            t.child_frame_id = child
            t.transform.translation.z = float(z)
            t.transform.rotation.w = 1.0
            tfs.append(t)
        self.static_tf_pub.sendTransform(tfs)

    def _publish_odom(self, o, stamp):
        qx, qy, qz, qw = yaw_to_quat(o['yaw'])
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = o['x']
        t.transform.translation.y = o['y']
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_pub.sendTransform(t)

        m = Odometry()
        m.header.stamp = stamp
        m.header.frame_id = self.odom_frame
        m.child_frame_id = self.base_frame
        m.pose.pose.position.x = o['x']
        m.pose.pose.position.y = o['y']
        m.pose.pose.orientation.z = qz
        m.pose.pose.orientation.w = qw
        m.twist.twist.linear.x = o['vx']
        m.twist.twist.angular.z = o['wz']
        # simulator odometry is exact; small covariances keep consumers happy
        for i in (0, 7, 14, 21, 28, 35):
            m.pose.covariance[i] = 1e-4
            m.twist.covariance[i] = 1e-4
        self.odom_pub.publish(m)

    def _publish_scan(self, ranges, stamp):
        n = len(ranges)
        if n == 0:
            return
        s = LaserScan()
        s.header.stamp = stamp
        s.header.frame_id = self.laser_frame
        s.angle_increment = 2.0 * math.pi / n
        s.angle_min = -math.pi
        s.angle_max = -math.pi + (n - 1) * s.angle_increment
        s.time_increment = 0.0
        s.scan_time = self.dt
        s.range_min = 0.1
        s.range_max = float(self.info['max_range'])
        s.ranges = ranges.tolist()
        self.scan_pub.publish(s)

    def _publish_rgb(self, rgb_bytes, stamp):
        w, h = int(self.info['rgb_width']), int(self.info['rgb_height'])
        if len(rgb_bytes) != w * h * 3:
            return
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self.camera_frame
        img.width, img.height = w, h
        img.encoding = 'rgb8'
        img.step = w * 3
        img.data = rgb_bytes
        self.img_pub.publish(img)

    def destroy_node(self):
        self.sock.close(0)
        self.ctx.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HabitatBridge()
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
