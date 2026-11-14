#!/usr/bin/env python3
# Open-vocabulary segmentation for the semantic mode: camera/image_raw -> sam3_server (ZMQ) ->
# camera/semantic/image_raw (mono8 class codes, see semantic.py), stamped like the RGB image so the
# voxel map pairs it with the depth frame of the same time. The drop-in replacement for
# habitat_server --semantic (ground truth).
#
# Only the newest image is sent; images arriving while the server is busy (~0.2 s per image with
# three prompts on an RTX 5070 Ti) are dropped. The request runs in a thread of its own so the
# image callback never waits for the server.

import json
import threading

import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import Image


class SemanticSegNode(Node):

    def __init__(self):
        super().__init__('semantic_seg')
        p = self.declare_parameter
        self.address = p('zmq_address', 'tcp://127.0.0.1:5557').value
        self.timeout_ms = int(1000 * float(p('timeout', 5.0).value))
        self.min_period = float(p('min_period', 0.0).value)     # s of sim time between requests
        self.pub = self.create_publisher(Image, p('semantic_topic', 'camera/semantic/image_raw').value, 2)
        self.create_subscription(Image, p('image_topic', 'camera/image_raw').value, self._on_image, 2)
        self.ctx = zmq.Context()
        self.sock = None
        self._connect()
        self.latest = None
        self.cv = threading.Condition()
        self.n, self.t_sum, self.last_stamp = 0, 0.0, None
        threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info(f'segmenting camera images with sam3_server at {self.address}')

    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

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
            self._segment(msg)

    def _segment(self, msg):
        w, h = msg.width, msg.height
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, -1)[:, :3 * w]
        self.sock.send_multipart([json.dumps({'op': 'segment', 'width': w, 'height': h}).encode(),
                                  np.ascontiguousarray(rgb).tobytes()])
        if not self.sock.poll(self.timeout_ms):
            self.get_logger().warn(f'no answer from sam3_server at {self.address} - is it running? retrying...',
                                   throttle_duration_sec=5.0)
            self._connect()           # a REQ socket cannot send again until it got a reply
            return
        parts = self.sock.recv_multipart()
        hdr = json.loads(parts[0].decode())
        if 'error' in hdr:
            self.get_logger().error(f"sam3_server error: {hdr['error']}", throttle_duration_sec=5.0)
            return
        out = Image(header=msg.header, height=h, width=w, encoding='mono8', step=w)
        out.data = parts[1]
        self.pub.publish(out)
        self.n += 1
        self.t_sum += hdr['ms']
        if self.n % 50 == 0:
            self.get_logger().info(f'[sam3] {self.n} images segmented, {self.t_sum / self.n:.0f} ms/image')


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegNode()
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
