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
#   request : [json {"op": "segment", "width": w, "height": h, "detect": {query: prompt}},
#              rgb uint8 bytes (h x w x 3)]
#             | [json {"op": "info"}]
#   reply   : [json {"ms": inference time, "scores": {class: [mask scores]},
#                    "detections": [{"query": q, "score": s}]},
#              class uint8 bytes (h x w), instance masks (np.packbits, ceil(h w / 8) bytes each)]
# Class codes as habitat_server --semantic (trav_exploration/semantic.py):
#   0 other, 1 floor, 2 rug, 3 stairs. One text prompt per class; where masks overlap, stairs win
#   over rug and rug over floor. Masks of the class image need --confidence.
# "detect" (optional): the instance map's queries (Sec. V-A, Q_obj and Q_land, e.g. a target
# object and "stairs"). Every instance mask SAM3 returns for such a query above
# --detect-confidence is a detection d_q = (R_q, s_q, q); a query named like a class ("stairs")
# reuses that class's prompt and inference.

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
    p.add_argument('--confidence', type=float, default=0.5, help='SAM3 mask score threshold (class image)')
    # No threshold on what counts as evidence (Sec. V-A): every detection is an observation for
    # the instance map, weighted there by its own score (InstanceMap score_weight). Only SAM3's
    # noise floor is cut: it answers every prompt with all of its 200 object queries, each with a
    # non-empty mask; below ~0.05 they are empty slots, not weak detections (a frame of HM3D 00821:
    # 200 masks of 650-1650 px for 'tv' with no TV in view, all scores <= 0.03).
    p.add_argument('--detect-confidence', type=float, default=0.05,
                   help='noise floor of a detection for the instance map (0.05-0.3: very low-confidence observations)')
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
        self.cls_conf = args.confidence
        self.proc = Sam3Processor(build_sam3_image_model(),
                                  confidence_threshold=min(args.confidence, args.detect_confidence))
        # floor first: later classes overwrite earlier ones
        self.prompts = [('floor', args.prompt_floor), ('rug', args.prompt_rug), ('stairs', args.prompt_stairs)]
        print(f'[sam3_server] model loaded in {time.time() - t0:.1f} s, GPU memory '
              f'{torch.cuda.memory_allocated() / 1e9:.1f} GB, prompts {dict(self.prompts)}')

    def segment(self, rgb, detect=None):
        """Class image, mask scores per class, and the detections of the `detect` queries
        [(query, score, bool mask)]."""
        from PIL import Image
        state = self.proc.set_image(Image.fromarray(rgb))
        cls = np.zeros(rgb.shape[:2], dtype=np.uint8)
        scores = {}
        dets = []
        detect = dict(detect or {})
        extra = [(q, p) for q, p in detect.items() if q not in CLASSES]
        for name, prompt in self.prompts + extra:
            self.proc.reset_all_prompts(state)
            out = self.proc.set_text_prompt(state=state, prompt=prompt)
            sc = out['scores'].float().cpu().numpy().reshape(-1)
            masks = out['masks'].reshape(-1, *cls.shape)
            if name in CLASSES:
                keep = sc >= self.cls_conf
                scores[name] = [round(float(v), 3) for v in sc[keep]]
                if keep.any():
                    m = masks[self.torch.as_tensor(keep, device=masks.device)].any(dim=0).cpu().numpy()
                    cls[m] = CLASSES[name]
            if name in detect:
                for i in range(len(sc)):
                    dets.append((name, float(sc[i]), masks[i].cpu().numpy()))
        return cls, scores, dets


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
            cls, scores, dets = seg.segment(rgb, req.get('detect'))
            ms = 1e3 * (time.time() - t0)
            hdr = {'ms': ms, 'scores': scores, 'detections': [{'query': q, 'score': round(s, 4)} for q, s, _ in dets]}
            packed = b''.join(np.packbits(m.reshape(-1)).tobytes() for _, _, m in dets)
            sock.send_multipart([json.dumps(hdr).encode(), cls.tobytes(), packed])
            n += 1
            t_sum += ms
            if time.time() - t_report > 10.0:
                print(f'[sam3_server] {n} images, {t_sum / n:.0f} ms/image')
                t_report = time.time()
        except Exception as e:  # always answer, or the REQ socket on the ROS side locks up
            sock.send_multipart([json.dumps({'error': repr(e)}).encode(), b''])
            if 'CUDA error' in repr(e):
                # the CUDA context is lost: every later request fails the same way (one 00820
                # run got no segmentation at all), so stop with a message instead of serving errors
                print(f'[sam3_server] {e!r}\n[sam3_server] CUDA context lost: exiting, start the server again',
                      flush=True)
                raise SystemExit(1)
    print(f'[sam3_server] done, {n} images')


if __name__ == '__main__':
    main()
