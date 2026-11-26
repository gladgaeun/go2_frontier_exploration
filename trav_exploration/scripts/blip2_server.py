#!/usr/bin/env python3
# BLIP-2 image-text matching server for the spatial value map of trav_exploration
# (TravExplorer Sec. V-B: "we use BLIP-2 as a fast image-text matching model", following VLFM).
# Runs in its own conda environment (blip2: torch + transformers); it does NOT import ROS.
# itm_node (ROS) sends it the camera images.
#
#   conda activate blip2
#   python -u blip2_server.py                   # tcp://127.0.0.1:5558
#
# Protocol (ZMQ REQ/REP, multipart):
#   request : [json {"op": "score", "width": w, "height": h, "texts": [text, ...]}, rgb uint8 bytes (h x w x 3)]
#             | [json {"op": "itc", "width": w, "height": h, "texts": [text, ...]}, rgb uint8 bytes]
#             | [json {"op": "info"}]
#   reply   : [json {"ms": inference time, "scores": [s per text]}]
# op "itc" (floor_switch node, room-type similarities): the raw ITC cosine similarity of the image
# to each text, max over the 32 query embeddings, not clipped, whatever --head is. The image is
# resized to 224 x 224 (PIL bicubic) and normalised as floor_switch/scripts/embed_frames.py does
# offline; the text embeddings are cached per text list. "score" requests are unchanged.
# Score (--head): 'itm' = probability of the image-text matching head that the text matches the
# image (in [0, 1]); 'itc' = cosine similarity of the image and text embeddings (VLFM's value),
# clipped to [0, 1].

import argparse
import json
import signal
import time

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='BLIP-2 image-text matching server')
    p.add_argument('--bind', default='tcp://127.0.0.1:5558')
    p.add_argument('--model', default='Salesforce/blip2-itm-vit-g')
    p.add_argument('--head', choices=['itm', 'itc'], default='itm')
    return p.parse_args(argv)


class Matcher:

    def __init__(self, args):
        import torch
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor
        self.torch = torch
        self.head = args.head
        t0 = time.time()
        self.proc = Blip2Processor.from_pretrained(args.model)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(args.model, torch_dtype=torch.float16).to('cuda').eval()
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device='cuda').view(1, 3, 1, 1)
        self.size = (ip.size['width'], ip.size['height'])
        self.text_emb = {}
        print(f'[blip2_server] {args.model} loaded in {time.time() - t0:.1f} s, head {self.head}, GPU memory '
              f'{torch.cuda.memory_allocated() / 1e9:.1f} GB', flush=True)

    def score(self, rgb, texts):
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(rgb)
        out = []
        with torch.inference_mode():
            for text in texts:
                inputs = self.proc(images=img, text=text, return_tensors='pt').to('cuda', torch.float16)
                if self.head == 'itm':
                    logits = self.model(**inputs, use_image_text_matching_head=True).logits_per_image
                    out.append(float(logits.float().softmax(dim=1)[0, 1]))
                else:
                    sim = self.model(**inputs, use_image_text_matching_head=False).logits_per_image
                    out.append(float(np.clip(sim.float().reshape(-1)[0].item(), 0.0, 1.0)))
        return out

    def _itc_text(self, texts):
        key = tuple(texts)
        if key not in self.text_emb:
            torch, m = self.torch, self.model
            with torch.inference_mode():
                tok = self.proc.tokenizer(list(texts), padding=True, return_tensors='pt').to('cuda')
                q = m.embeddings(input_ids=tok['input_ids'])
                out = m.qformer(query_embeds=q, query_length=0,
                                attention_mask=tok['attention_mask']).last_hidden_state
                out = out.to(m.text_projection.weight.dtype)
                self.text_emb[key] = torch.nn.functional.normalize(m.text_projection(out[:, 0, :]), dim=-1)
        return self.text_emb[key]

    def itc(self, rgb, texts):
        """Raw ITC similarity per text (max over the query embeddings), as embed_frames.py."""
        from PIL import Image
        torch, m = self.torch, self.model
        temb = self._itc_text(texts)
        img = np.asarray(Image.fromarray(rgb).resize(self.size, Image.BICUBIC))
        with torch.inference_mode():
            x = torch.from_numpy(np.array(img)[None]).to('cuda').permute(0, 3, 1, 2).float() / 255.0
            x = ((x - self.mean) / self.std).half()
            v = m.vision_model(pixel_values=x).last_hidden_state
            qt = m.query_tokens.expand(v.shape[0], -1, -1)
            mask = torch.ones(v.shape[:2], dtype=torch.long, device='cuda')
            o = m.qformer(query_embeds=qt, encoder_hidden_states=v, encoder_attention_mask=mask).last_hidden_state
            o = o.to(m.vision_projection.weight.dtype)
            e = torch.nn.functional.normalize(m.vision_projection(o), dim=-1)      # (1, 32, 256)
            sims = (e.float() @ temb.float().T).max(dim=1).values[0]
        return [float(v) for v in sims.cpu().numpy()]


def main():
    args = parse_args()
    import zmq

    m = Matcher(args)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f'[blip2_server] listening on {args.bind}  (Ctrl+C to quit)', flush=True)

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
                sock.send_multipart([json.dumps({'head': m.head}).encode()])
                continue
            w, h = int(req['width']), int(req['height'])
            rgb = np.frombuffer(parts[1], dtype=np.uint8).reshape(h, w, 3)
            t0 = time.time()
            if req.get('op') == 'itc':
                sims = m.itc(rgb, req['texts'])
                sock.send_multipart([json.dumps({'ms': 1e3 * (time.time() - t0), 'scores': sims}).encode()])
                continue
            scores = m.score(rgb, req['texts'])
            ms = 1e3 * (time.time() - t0)
            sock.send_multipart([json.dumps({'ms': ms, 'scores': scores}).encode()])
            n += 1
            t_sum += ms
            if time.time() - t_report > 10.0:
                print(f'[blip2_server] {n} images, {t_sum / n:.0f} ms/image, last {scores}', flush=True)
                t_report = time.time()
        except Exception as e:  # always answer, or the REQ socket on the ROS side locks up
            sock.send_multipart([json.dumps({'error': repr(e)}).encode()])
            if 'CUDA error' in repr(e):
                # the CUDA context is lost: every later request fails the same way (one 00820
                # run got no segmentation at all), so stop with a message instead of serving errors
                print(f'[blip2_server] {e!r}\n[blip2_server] CUDA context lost: exiting, start the server again',
                      flush=True)
                raise SystemExit(1)
    print(f'[blip2_server] done, {n} images', flush=True)


if __name__ == '__main__':
    main()
