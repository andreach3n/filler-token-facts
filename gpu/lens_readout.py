"""Phase 2: read token predictions out of the stored hidden states with three lenses.

For every state file from extract_states.py (one prompt; [layers, positions, d_model]), and for
each lens, compute a probability distribution over the vocabulary at every (layer, position):

  logit lens : softmax(W_U . norm(h))
  J-lens     : softmax(W_U . norm(J_l . h))      (Jacobian lens, camilablank/workspace-lenses)
  R-lens     : same with the RelP-fitted Jacobians

Then apply paper 1's cross-example residual: within a group (condition, statement set), subtract
the per-(layer, position) mean distribution over the group's prompts. That removes what every
prompt shares at a position (the filler text itself, formatting) and leaves what is specific to
this prompt, which is where intermediate values show up.

Two passes per group: pass A accumulates the group mean on the GPU, pass B recomputes each
prompt's distributions and subtracts it. Per prompt we save, for each lens/layer/position: the
top-10 raw tokens, the top-10 residual tokens, and the raw/residual values of the target tokens
(the problem's intermediates x, c1x, y, c2y and the answer).

  python lens_readout.py --device cuda                       # full run
  python lens_readout.py --device cpu --layers 30 41 --limit 3   # quick check on a few files
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

STATES = Path("/workspace/states")
OUT = Path("/workspace/readouts")
HEAD = Path("/workspace/head.pt")
LENSES = {
    "logit": None,
    "jlens": Path("/workspace/lens/deepseek-v4-flash/j-lens/lens.pt"),
    "rlens": Path("/workspace/lens/deepseek-v4-flash/r-lens/lens.pt"),
}
TARGETS = ("x", "c1x", "y", "c2y", "answer")
TOPK = 10
RMS_EPS = 1e-6


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("/dev/shm/models/V4F-mp2")


def target_token_ids(tok, intermediates):
    """Token id of each intermediate value; DeepSeek keeps numbers 0-999 as single tokens."""
    ids = []
    for name in TARGETS:
        enc = tok.encode(str(intermediates[name]), add_special_tokens=False)
        ids.append(enc[0] if len(enc) == 1 else -1)
    return ids


class Readout:
    def __init__(self, device, layers):
        head = torch.load(HEAD, map_location=device)
        self.W = head["head"].to(device, torch.float16 if device != "cpu" else torch.float32)
        self.norm_w = head["norm"].to(device, torch.float32)
        self.device = device
        self.layers = layers
        self.J = {}
        for name, path in LENSES.items():
            if path is None:
                continue
            lens = torch.load(path, map_location="cpu", weights_only=True)
            self.J[name] = {l: lens["J"][l].to(device, self.W.dtype) for l in layers if l in lens["J"]}

    def probs(self, states, lens):
        """states [L, P, d] -> probabilities [L, P, V] (layers missing from a lens give zeros)."""
        h = states.to(self.device, self.W.dtype)
        if lens != "logit":
            J = self.J[lens]
            h = torch.stack([h[i] @ J[l].T if l in J else torch.zeros_like(h[i]) for i, l in enumerate(self.layers)])
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * self.norm_w
        logits = x.to(self.W.dtype) @ self.W.T
        return logits.float().softmax(-1)


def group_key(meta):
    return f"{meta['condition']}__{meta['set']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="subset of layers (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="files per group (for quick checks)")
    parser.add_argument("--groups", nargs="*", default=None, help="only these condition__set groups")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()

    files = sorted(STATES.glob("*.npz"))
    groups = defaultdict(list)
    for f in files:
        meta = json.loads(str(np.load(f)["meta"]))
        groups[group_key(meta)].append(f)
    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}
    if args.limit:
        groups = {k: v[: args.limit] for k, v in groups.items()}

    first = np.load(files[0])
    all_layers = [int(l) for l in first["layers"]]
    layers = args.layers or all_layers
    layer_idx = [all_layers.index(l) for l in layers]
    reader = Readout(args.device, layers)
    print(f"{len(files)} state files in {len(groups)} groups; lenses {list(LENSES)}; layers {layers[:3]}...{layers[-3:]}")

    for key, members in groups.items():
        pending = [f for f in members if not (OUT / f.name).exists()]
        if not pending:
            print(f"group {key}: all {len(members)} done")
            continue
        # Pass A: group mean per lens over all members (not just pending ones).
        mean = {}
        n_pos = None
        for f in members:
            d = np.load(f)
            states = torch.from_numpy(d["states"][layer_idx])
            n_pos = n_pos or states.shape[1]
            assert states.shape[1] == n_pos, f"position count differs within group {key}: {f.name}"
            for lens in LENSES:
                p = reader.probs(states, lens)
                mean[lens] = p if lens not in mean else mean[lens] + p
        for lens in mean:
            mean[lens] /= len(members)
        print(f"group {key}: mean over {len(members)} prompts, {n_pos} positions")

        # Pass B: per-prompt raw and residual readouts.
        for f in pending:
            d = np.load(f)
            states = torch.from_numpy(d["states"][layer_idx])
            targets = torch.tensor(target_token_ids(tok, json.loads(str(d["intermediates"]))), device=args.device)
            out = {}
            for lens in LENSES:
                p = reader.probs(states, lens)
                r = p - mean[lens]
                raw_top = p.topk(TOPK, dim=-1)
                res_top = r.topk(TOPK, dim=-1)
                valid = targets >= 0
                tgt = targets.clamp(min=0)
                out[f"{lens}_raw_top_ids"] = raw_top.indices.cpu().numpy().astype(np.int32)
                out[f"{lens}_raw_top_p"] = raw_top.values.cpu().numpy().astype(np.float16)
                out[f"{lens}_res_top_ids"] = res_top.indices.cpu().numpy().astype(np.int32)
                out[f"{lens}_res_top_v"] = res_top.values.cpu().numpy().astype(np.float16)
                out[f"{lens}_target_raw_p"] = torch.where(valid, p[..., tgt], torch.nan).cpu().numpy().astype(np.float32)
                out[f"{lens}_target_res_v"] = torch.where(valid, r[..., tgt], torch.nan).cpu().numpy().astype(np.float32)
                # Rank of each target token under the residual (0 = top), the metric paper 1 reports.
                # One target at a time keeps the comparison at [L, P, V] instead of [L, P, T, V].
                ranks = torch.stack([(r > r[..., t : t + 1]).sum(-1) for t in tgt.tolist()], dim=-1)
                out[f"{lens}_target_res_rank"] = torch.where(valid, ranks, -1).cpu().numpy().astype(np.int32)
            np.savez(
                OUT / f.name, layers=np.array(layers, dtype=np.int32), positions=d["positions"], n_filler=d["n_filler"],
                targets=np.array(TARGETS), target_ids=targets.cpu().numpy().astype(np.int32),
                answer_prob=d["answer_prob"], greedy_token_id=d["greedy_token_id"], answer_token_id=d["answer_token_id"],
                meta=d["meta"], intermediates=d["intermediates"], **out,
            )
        print(f"group {key}: wrote {len(pending)} readouts")


if __name__ == "__main__":
    main()
