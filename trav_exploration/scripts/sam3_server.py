#!/usr/bin/env python3
# SAM3 open-vocabulary segmentation server for the semantic traversability of trav_exploration
# (TravExplorer Sec. III-A: "given robot-traversable categories (e.g. floor, stair, and rug), the
# RGB image is segmented to extract traversable regions").
# Runs in its own conda environment (SAM3 needs Python >= 3.12); it does NOT import ROS.
# semantic_seg_node (ROS) sends it the camera images.
#
#   conda activate sam3
#   python sam3_server.py                       # tcp://127.0.0.1:5557
#
# Protocol (ZMQ REQ/REP, multipart):
#   request : [json {"op": "segment", "width": w, "height": h}, rgb uint8 bytes (h x w x 3)]
#             | [json {"op": "info"}]
#   reply   : [json {"ms": inference time, "scores": {class: [mask scores]}}, class uint8 bytes (h x w)]
# Class codes as habitat_server --semantic (trav_exploration/semantic.py):
#   0 other, 1 floor, 2 rug, 3 stairs. One text prompt per class; where masks overlap, stairs win
#   over rug and rug over floor.

import argparse
import json
import signal
import time

import numpy as np

CLASSES = {'floor': 1, 'rug': 2, 'stairs': 3}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='SAM3 segmentation server (text prompts -> class image)')
    p.add_argument('--bind', default='tcp://127.0.0.1:5557')
    p.add_argument('--prompt-floor', default='floor')
    p.add_argument('--prompt-rug', default='rug')
    p.add_argument('--prompt-stairs', default='stairs')
    p.add_argument('--confidence', type=float, default=0.5, help='SAM3 mask score threshold')
    return p.parse_args(argv)


class Segmenter:

    def __init__(self, args):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.torch = torch
        # as in the SAM3 examples: the model runs in bfloat16 (fp32 inputs fail on dtype)
        torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
        t0 = time.time()
        self.proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=args.confidence)
        # floor first: later classes overwrite earlier ones
        self.prompts = [('floor', args.prompt_floor), ('rug', args.prompt_rug), ('stairs', args.prompt_stairs)]
        print(f'[sam3_server] model loaded in {time.time() - t0:.1f} s, GPU memory '
              f'{torch.cuda.memory_allocated() / 1e9:.1f} GB, prompts {dict(self.prompts)}')

    def segment(self, rgb):
        from PIL import Image
        state = self.proc.set_image(Image.fromarray(rgb))
        cls = np.zeros(rgb.shape[:2], dtype=np.uint8)
        scores = {}
        for name, prompt in self.prompts:
            self.proc.reset_all_prompts(state)
            out = self.proc.set_text_prompt(state=state, prompt=prompt)
            masks = out['masks']
            scores[name] = [round(float(s), 3) for s in out['scores']]
            if len(scores[name]):
                m = masks.reshape(-1, *cls.shape).any(dim=0).cpu().numpy()
                cls[m] = CLASSES[name]
        return cls, scores


def main():
    args = parse_args()
    import zmq

    seg = Segmenter(args)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f'[sam3_server] listening on {args.bind}  (Ctrl+C to quit)')

    running = True

    def _stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    n, t_sum, t_report = 0, 0.0, time.time()
    while running:
        if not poller.poll(200):
            continue
        parts = sock.recv_multipart()
        try:
            req = json.loads(parts[0].decode())
            if req.get('op') == 'info':
                sock.send_multipart([json.dumps({'classes': CLASSES, 'prompts': dict(seg.prompts)}).encode(), b''])
                continue
            w, h = int(req['width']), int(req['height'])
            rgb = np.frombuffer(parts[1], dtype=np.uint8).reshape(h, w, 3)
            t0 = time.time()
            cls, scores = seg.segment(rgb)
            ms = 1e3 * (time.time() - t0)
            sock.send_multipart([json.dumps({'ms': ms, 'scores': scores}).encode(), cls.tobytes()])
            n += 1
            t_sum += ms
            if time.time() - t_report > 10.0:
                print(f'[sam3_server] {n} images, {t_sum / n:.0f} ms/image')
                t_report = time.time()
        except Exception as e:  # always answer, or the REQ socket on the ROS side locks up
            sock.send_multipart([json.dumps({'error': repr(e)}).encode(), b''])
    print(f'[sam3_server] done, {n} images')


if __name__ == '__main__':
    main()
