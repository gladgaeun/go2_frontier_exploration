#!/usr/bin/env python3
"""BLIP-2 ITC embeddings and room similarities of the rendered frames (render_frames.py).

For each <frames>/rgb/<id>.npy (224 x 224 RGB, already BLIP-2's input size):
  <out>/<id>.npz  t, emb (N, 32, 256) float16: the normalised ITC image embeddings (32 queries),
                  sims (N, 9) float32: raw cosine similarity to each room prompt (floor_switch.rooms),
                  max over the 32 queries; no temperature, no softmax
and removes the rgb file. Runs alongside render_frames.py (waits for new files until its
render_done marker). Same computation as Blip2ForImageTextRetrieval(use_image_text_matching_head=False).
Run in the BLIP-2 env (e.g. `conda activate blip2`).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from floor_switch.rooms import PROMPTS  # noqa: E402


class ITC:
    def __init__(self, name):
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor
        self.proc = Blip2Processor.from_pretrained(name)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(name, torch_dtype=torch.float16).to('cuda').eval()
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device='cuda').view(1, 3, 1, 1)
        assert ip.size['height'] == 224 and ip.size['width'] == 224

    @torch.inference_mode()
    def text(self, prompts):
        m = self.model
        tok = self.proc.tokenizer(list(prompts), padding=True, return_tensors='pt').to('cuda')
        q = m.embeddings(input_ids=tok['input_ids'])
        out = m.qformer(query_embeds=q, query_length=0, attention_mask=tok['attention_mask']).last_hidden_state
        out = out.to(m.text_projection.weight.dtype)
        return torch.nn.functional.normalize(m.text_projection(out[:, 0, :]), dim=-1)

    @torch.inference_mode()
    def image(self, rgb):
        m = self.model
        x = torch.from_numpy(rgb).to('cuda').permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).half()
        v = m.vision_model(pixel_values=x).last_hidden_state
        q = m.query_tokens.expand(v.shape[0], -1, -1)
        mask = torch.ones(v.shape[:2], dtype=torch.long, device='cuda')
        o = m.qformer(query_embeds=q, encoder_hidden_states=v, encoder_attention_mask=mask).last_hidden_state
        o = o.to(m.vision_projection.weight.dtype)
        return torch.nn.functional.normalize(m.vision_projection(o), dim=-1)


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', default=H('~/objectnav_runs/floor_switch/frames'))
    ap.add_argument('--out', default=H('~/objectnav_runs/floor_switch/itc'))
    ap.add_argument('--model', default='Salesforce/blip2-itm-vit-g')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--keep-rgb', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    itc = ITC(args.model)
    temb = itc.text(PROMPTS)                                  # (9, 256)
    np.save(os.path.join(args.out, 'text_emb.npy'), temb.float().cpu().numpy())
    with open(os.path.join(args.out, 'prompts.txt'), 'w') as f:
        f.write('\n'.join(PROMPTS) + '\n')
    rgb_dir = os.path.join(args.frames, 'rgb')
    n = 0
    while True:
        files = sorted(f for f in os.listdir(rgb_dir) if f.endswith('.npy') and not f.endswith('.tmp.npy')
                       and not os.path.exists(os.path.join(args.out, f[:-4] + '.npz')))
        if not files:
            if os.path.exists(os.path.join(args.frames, 'render_done')):
                break
            time.sleep(2.0)
            continue
        for fn in files:
            eid = fn[:-4]
            t0 = time.time()
            rgb = np.load(os.path.join(rgb_dir, fn))
            embs, sims = [], []
            for i in range(0, len(rgb), args.batch):
                e = itc.image(rgb[i:i + args.batch])           # (b, 32, 256)
                embs.append(e.half().cpu().numpy())
                sims.append((e.float() @ temb.float().T).max(dim=1).values.cpu().numpy())
            g = np.load(os.path.join(args.frames, 'geom', eid + '.npz'))
            np.savez(os.path.join(args.out, eid + '.npz'), t=g['t'], emb=np.concatenate(embs),
                     sims=np.concatenate(sims).astype(np.float32))
            if not args.keep_rgb:
                os.remove(os.path.join(rgb_dir, fn))
            n += 1
            print(f'[{n}] {eid}: {len(rgb)} frames, {1e3 * (time.time() - t0) / len(rgb):.1f} ms/frame', flush=True)


if __name__ == '__main__':
    main()
