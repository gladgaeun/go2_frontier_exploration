#!/usr/bin/env python3
# Image-text relevance for the spatial value map (TravExplorer Sec. V-B): camera/image_raw ->
# blip2_server (ZMQ) -> camera/itm_score (trav_interfaces/ImageScore), stamped like the image so
# the voxel map projects it with the depth frame of the same time.
#
# The text is prompt.format(target=target) (VLFM's prompt by default). Only the newest image is
# sent; images arriving while the server is busy are dropped. The request runs in a thread of
# its own so the image callback never waits for the server.
#
# camera/itm_image (rgb8, for RViz): the scored image tinted with the colour of its score (the
# rainbow of the VoxelValue display, 0 red ... 1 magenta), each column as opaque as its
# observation weight w = cos^2(pi theta / theta_FOV) in eq. 7 (the white curve), with the score,
# the text and a colour bar. BLIP-2 gives one score per image, not per pixel: the tint shows how
# much of it each column passes to the traversable voxels it sees.

import json
import math
import threading

import cv2

import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from trav_exploration.value_map import rviz_rainbow, score_overlay
from trav_interfaces.msg import ImageScore


class ItmNode(Node):

    def __init__(self):
        super().__init__('itm')
        p = self.declare_parameter
        self.address = p('zmq_address', 'tcp://127.0.0.1:5558').value
        self.timeout_ms = int(1000 * float(p('timeout', 5.0).value))
        self.min_period = float(p('min_period', 0.0).value)     # s of sim time between requests
        target = str(p('target', '').value).split('=', 1)
        self.text = str(p('prompt', 'Seems like there is a {target} ahead.').value).format(
            target=target[-1].strip())
        self.pub = self.create_publisher(ImageScore, p('score_topic', 'camera/itm_score').value, 5)
        self.create_subscription(Image, p('image_topic', 'camera/image_raw').value, self._on_image, 2)
        self.vis_pub = self.create_publisher(Image, p('vis_topic', 'camera/itm_image').value, 2) \
            if p('visualize', True).value else None
        self.intr = None                                         # (fx, cx, hfov) from camera_info
        self.create_subscription(CameraInfo, p('camera_info_topic', 'camera/camera_info').value, self._on_info, 1)
        self.ctx = zmq.Context()
        self.sock = None
        self._connect()
        self.latest = None
        self.cv = threading.Condition()
        self.n, self.t_sum, self.s_max, self.last_stamp = 0, 0.0, 0.0, None
        threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info(f'image-text matching with blip2_server at {self.address}: "{self.text}"')

    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

    def _on_info(self, msg):
        fx, cx = msg.k[0], msg.k[2]
        if fx > 0.0:
            self.intr = (fx, cx, 2.0 * math.atan(0.5 * msg.width / fx))

    def _visualize(self, msg, rgb, s):
        if self.vis_pub is None or self.intr is None or self.vis_pub.get_subscription_count() == 0:
            return
        h, w = msg.height, msg.width
        rgb = np.asarray(rgb).reshape(h, w, 3)
        img, wt = score_overlay(rgb, s, *self.intr)
        img = np.ascontiguousarray(img)
        # observation weight per column: white curve along the bottom (w = 1 at 25 % height)
        ys = (h - 1 - 0.25 * h * wt).astype(np.int32)
        cv2.polylines(img, [np.stack([np.arange(w), ys], axis=1).astype(np.int32)], False, (255, 255, 255), 2)
        # colour bar 0..1 on the right with a mark at the score
        x0, y0, bh = w - 22, 40, h - 80
        bar = (255 * rviz_rainbow(1.0 - np.linspace(0, 1, bh))).astype(np.uint8)
        img[y0:y0 + bh, x0:x0 + 14] = bar[:, None, :]
        ym = int(y0 + (1.0 - min(max(s, 0.0), 1.0)) * (bh - 1))
        cv2.line(img, (x0 - 8, ym), (x0 + 14, ym), (255, 255, 255), 3)
        cv2.putText(img, '1', (x0 - 14, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, '0', (x0 - 14, y0 + bh), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(img, f'BLIP-2 ITM {s:.3f} (max {self.s_max:.3f})', (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, self.text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        out = Image(header=msg.header, height=h, width=w, encoding='rgb8', is_bigendian=0, step=3 * w)
        out.data = img.tobytes()
        self.vis_pub.publish(out)

    def _on_image(self, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().error(f'expected rgb8, got {msg.encoding}', throttle_duration_sec=10.0)
            return
        with self.cv:
            self.latest = msg
            self.cv.notify()

    def _worker(self):
        while rclpy.ok():
            with self.cv:
                while self.latest is None and rclpy.ok():
                    self.cv.wait(0.5)
                msg, self.latest = self.latest, None
            if msg is None:
                continue
            t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            if self.last_stamp is not None and 0.0 <= t - self.last_stamp < self.min_period:
                continue
            self.last_stamp = t
            self._score(msg)

    def _score(self, msg):
        w, h = msg.width, msg.height
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, -1)[:, :3 * w]
        self.sock.send_multipart([json.dumps({'op': 'score', 'width': w, 'height': h, 'texts': [self.text]}).encode(),
                                  np.ascontiguousarray(rgb).tobytes()])
        if not self.sock.poll(self.timeout_ms):
            self.get_logger().warn(f'no answer from blip2_server at {self.address} - is it running? retrying...',
                                   throttle_duration_sec=5.0)
            self._connect()           # a REQ socket cannot send again until it got a reply
            return
        hdr = json.loads(self.sock.recv_multipart()[0].decode())
        if 'error' in hdr:
            self.get_logger().error(f"blip2_server error: {hdr['error']}", throttle_duration_sec=5.0)
            return
        s = float(hdr['scores'][0])
        self.pub.publish(ImageScore(header=msg.header, text=self.text, score=s))
        self.n += 1
        self.t_sum += hdr['ms']
        self.s_max = max(self.s_max, s)
        self._visualize(msg, rgb, s)
        if self.n % 50 == 0:
            self.get_logger().info(f'[itm] {self.n} images, {self.t_sum / self.n:.0f} ms/image, last score '
                                   f'{s:.3f}, max {self.s_max:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = ItmNode()
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
