"""Do the statement positions read and compute at the same time?

Uses only the stored phase-2 readouts (no model, no API). For each filler position and layer:
  reading   : is the raw (un-subtracted) logit-lens top-1 the actual next token of the filler text?
  computing : is the residual top-1 one of the computed intermediates (y, 2y, answer)?
  truth     : at the token before each statement's object ("... was directed by"), is the raw
              top-1 the TRUE object or the WRITTEN (false) one? Mean subtraction cannot see this,
              because every prompt in a set shares the statements; the raw readout can.
  roles     : computing rate by token role inside a sentence (object tokens, final period, other).

  python dual_use.py --readouts /root/readouts --out /root/results/dual_use.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ENC = "/dev/shm/models/DeepSeek-V4-Flash/encoding"
CKPT = "/dev/shm/models/V4F-mp2"
sys.path.insert(0, ENC)
from encoding_dsv4 import encode_messages  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

COMPUTED = (2, 3, 4)  # indices of y, c2y, answer in the stored targets (x, c1x, y, c2y, answer)


def mean_of(lists):
    return {k: float(np.mean(v)) for k, v in lists.items()} if isinstance(lists, dict) else float(np.mean(lists))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", type=Path, default=Path("/root/readouts"))
    parser.add_argument("--prompts", type=Path, default=Path("/workspace/gpu_prompts_soe300.jsonl"))
    parser.add_argument("--sets", type=Path, default=Path("/root/sets.json"))
    parser.add_argument("--out", type=Path, default=Path("/root/results/dual_use.json"))
    parser.add_argument("--lens", default="logit")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(CKPT)
    specs = {json.loads(l)["id"]: json.loads(l) for l in args.prompts.read_text().splitlines()}
    sets = {s["id"]: s["pairs"] for s in json.loads(args.sets.read_text())}
    lens = args.lens

    def first_id(text):
        return tok.encode(" " + text, add_special_tokens=False)[0]

    # accumulators: condition -> layer -> list
    reading = defaultdict(lambda: defaultdict(list))        # raw top-1 == next token
    p_next = defaultdict(lambda: defaultdict(list))         # raw p(next token) (0 if outside top-10)
    computing = defaultdict(lambda: defaultdict(list))      # residual top-1 in {y, 2y, answer}
    joint = defaultdict(lambda: defaultdict(lambda: {"read_given_comp": [], "read_given_nocomp": [],
                                                      "pnext_given_comp": [], "pnext_given_nocomp": []}))
    truth = defaultdict(lambda: defaultdict(lambda: {"top1_true": [], "top1_written": [], "p_true": [], "p_written": []}))
    roles = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cond -> role -> layer -> computing hits
    n_prompts = defaultdict(int)

    for f in sorted(args.readouts.glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        cond, sid = meta["condition"], meta["id"]
        if cond == "none" or int(d["greedy_token_id"]) != int(d["answer_token_id"]):
            continue
        spec = specs[sid]
        messages = [{"role": "system", "content": spec["system"]}] + spec["messages"]
        prompt = encode_messages(messages, thinking_mode="chat")
        enc = tok(prompt, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        positions = [int(p) for p in d["positions"]]
        n_fill = int(d["n_filler"])
        filler_pos = positions[:n_fill]
        layers = [int(l) for l in d["layers"]]
        raw_top = d[f"{lens}_raw_top_ids"][:, :n_fill]      # [L, P, 10]
        raw_p = d[f"{lens}_raw_top_p"][:, :n_fill].astype(np.float32)
        res_top = d[f"{lens}_res_top_ids"][:, :n_fill, 0]   # [L, P]
        targets = d["target_ids"]
        comp_ids = {int(targets[i]) for i in COMPUTED if targets[i] >= 0}
        n_prompts[cond] += 1

        next_ids = np.array([ids[p + 1] for p in filler_pos])                     # [P]
        read_hit = raw_top[:, :, 0] == next_ids[None, :]                          # [L, P]
        pn = np.where(raw_top == next_ids[None, :, None], raw_p, 0.0).sum(-1)     # [L, P]
        comp_hit = np.isin(res_top, list(comp_ids))                               # [L, P]
        for li, l in enumerate(layers):
            reading[cond][l].append(read_hit[li].mean())
            p_next[cond][l].append(pn[li].mean())
            computing[cond][l].append(comp_hit[li].mean())
            c, nc = comp_hit[li], ~comp_hit[li]
            if c.any():
                joint[cond][l]["read_given_comp"].append(read_hit[li][c].mean())
                joint[cond][l]["pnext_given_comp"].append(pn[li][c].mean())
            if nc.any():
                joint[cond][l]["read_given_nocomp"].append(read_hit[li][nc].mean())
                joint[cond][l]["pnext_given_nocomp"].append(pn[li][nc].mean())

        kind = cond.split("-")[0]  # "false-unique" -> "false"
        if kind in ("false", "true"):
            filler_start = prompt.rfind(spec["filler_text"])
            cursor = 0
            role = np.full(n_fill, "other", dtype=object)
            for pair in sets[meta["set"]]:
                sentence = pair["false_sentence"] if kind == "false" else pair["true_sentence"]
                written = pair["false_object"] if kind == "false" else pair["true_object"]
                alt = pair["true_object"] if kind == "false" else pair["false_object"]
                s_off = spec["filler_text"].index(sentence, cursor)
                cursor = s_off + len(sentence)
                o_start = filler_start + s_off + sentence.index(written)
                o_end = o_start + len(written)
                obj_idx = [k for k, p in enumerate(filler_pos) if offsets[p][0] < o_end and offsets[p][1] > o_start]
                period = [k for k, p in enumerate(filler_pos) if offsets[p][0] >= filler_start + cursor - 1 and offsets[p][1] <= filler_start + cursor]
                for k in obj_idx:
                    role[k] = "object"
                for k in period:
                    role[k] = "period"
                if not obj_idx or obj_idx[0] == 0:
                    continue
                k = obj_idx[0] - 1  # the position that predicts the object's first token
                w_id, a_id = first_id(written), first_id(alt)
                t_id = a_id if kind == "false" else w_id
                for li, l in enumerate(layers):
                    top, ps = raw_top[li, k], raw_p[li, k]
                    truth[cond][l]["top1_true"].append(top[0] == t_id)
                    truth[cond][l]["top1_written"].append(top[0] == w_id)
                    truth[cond][l]["p_true"].append(float(ps[top == t_id].sum()))
                    truth[cond][l]["p_written"].append(float(ps[top == w_id].sum()))
            for li, l in enumerate(layers):
                for r in ("object", "period", "other"):
                    m = role == r
                    if m.any():
                        roles[cond][r][l].append(comp_hit[li][m].mean())

    out = {"n_prompts": dict(n_prompts), "layers": layers,
           "reading": {c: {l: float(np.mean(v)) for l, v in d.items()} for c, d in reading.items()},
           "p_next": {c: {l: float(np.mean(v)) for l, v in d.items()} for c, d in p_next.items()},
           "computing": {c: {l: float(np.mean(v)) for l, v in d.items()} for c, d in computing.items()},
           "joint": {c: {l: {k: float(np.mean(v)) if v else None for k, v in dd.items()} for l, dd in d.items()} for c, d in joint.items()},
           "truth": {c: {l: {k: float(np.mean(v)) for k, v in dd.items()} for l, dd in d.items()} for c, d in truth.items()},
           "roles": {c: {r: {l: float(np.mean(v)) for l, v in dd.items()} for r, dd in d.items()} for c, d in roles.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    show = [l for l in (20, 25, 30, 33, 35, 37, 39, 41, 42) if l in layers]
    print(f"prompts (correct only): {dict(n_prompts)}\n")
    print("== READING (raw top-1 == next filler token) vs COMPUTING (residual top-1 in {y, 2y, answer}), mean over filler positions")
    conds = sorted(out["reading"])
    print("layer  " + "  ".join(f"{c:>24s}" for c in conds))
    for l in show:
        print(f"{l:5d}  " + "  ".join(f"{c[:8]:8s} read {out['reading'][c][l]:.2f} comp {out['computing'][c][l]:.3f}" for c in conds))
    print("\n== JOINT: P(reading intact) and raw p(next) at positions WITH vs WITHOUT an intermediate present")
    for c in conds:
        print(f"  {c}:")
        for l in [l for l in (30, 33, 35, 37, 39) if l in layers]:
            j = out["joint"][c][l]
            print(f"    L{l}: read|comp {j['read_given_comp'] if j['read_given_comp'] is None else round(j['read_given_comp'], 2)}  read|nocomp {round(j['read_given_nocomp'], 2)}"
                  f"   p_next|comp {j['pnext_given_comp'] if j['pnext_given_comp'] is None else round(j['pnext_given_comp'], 2)}  p_next|nocomp {round(j['pnext_given_nocomp'], 2)}")
    print("\n== TRUTH at the token before each object: raw top-1 == TRUE object vs == WRITTEN object (false condition: written is false)")
    for c in sorted(out["truth"]):
        print(f"  {c}:")
        for l in show:
            t = out["truth"][c][l]
            print(f"    L{l}: top1=true {t['top1_true']:.2f}  top1=written {t['top1_written']:.2f}  p(true) {t['p_true']:.2f}  p(written) {t['p_written']:.2f}")
    for c in sorted(out["roles"]):
        print(f"\n== COMPUTING rate by token role ({c})")
        for r in ("object", "period", "other"):
            print(f"  {r:7s} " + "  ".join(f"L{l}:{out['roles'][c][r][l]:.3f}" for l in (30, 33, 35, 37, 39) if l in layers and l in out['roles'][c][r]))


if __name__ == "__main__":
    main()
