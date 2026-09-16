"""Phase 3: aggregate lens readouts into per-condition decoding rates, with chance baselines.

For each prompt, lens and layer: is each target token (x, c1x, y, c2y, answer) the top-1 / in
the top-10 of the cross-example residual at any filler position, and at the answer position?
Averaged over prompts per condition, that is paper 1's "decodable intermediate" rate.

Two safeguards:
- Chance baseline: the same statistic computed with another problem's targets (the next prompt
  in the same condition). "Any of ~110 positions at the best of 43 layers" is loose, so a target
  only counts as decodable to the extent it beats this baseline.
- x is written in the prompt, so finding it at filler positions is copying; y, c1x, c2y and the
  answer never appear in the prompt, so those are the computation targets.

The J-lens and R-lens cover layers 0-41 only; layer 42 is masked for them.

  python aggregate.py --readouts /root/readouts --out /root/results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LENSES = ("logit", "jlens", "rlens")
LENS_MAX_LAYER = {"logit": 42, "jlens": 41, "rlens": 41}
CONDITIONS = ("none", "counting", "false", "true")
KS = (1, 10)


def load(readouts: Path):
    rows = []
    for f in sorted(readouts.glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        row = {
            "id": meta["id"], "condition": meta["condition"], "set": meta["set"], "problem_id": meta["problem_id"],
            "correct": int(d["greedy_token_id"]) == int(d["answer_token_id"]),
            "answer_prob": float(d["answer_prob"]),
            "n_filler": int(d["n_filler"]), "layers": [int(l) for l in d["layers"]],
            "targets": [str(t) for t in d["targets"]], "target_ids": d["target_ids"],
        }
        for lens in LENSES:
            row[f"{lens}_rank"] = d[f"{lens}_target_res_rank"]      # [L, P, T]
            row[f"{lens}_top"] = d[f"{lens}_res_top_ids"]           # [L, P, 10]
            row[f"{lens}_rawp"] = d[f"{lens}_target_raw_p"]         # [L, P, T]
        rows.append(row)
    return rows


def layer_mask(row, lens):
    return np.array([l <= LENS_MAX_LAYER[lens] for l in row["layers"]])[:, None, None]


def block(arr, row, role):
    return arr[:, : row["n_filler"]] if role == "filler" else arr[:, -1:]


def real_hits(row, lens, k, role):
    ranks = block(row[f"{lens}_rank"], row, role)
    return ((ranks >= 0) & (ranks < k)) & layer_mask(row, lens)


def chance_hits(row, other, lens, k, role):
    """Same statistic with another problem's target ids, via membership in the stored top-10."""
    top = block(row[f"{lens}_top"], row, role)[..., :k]              # [L, P, k]
    ids = other["target_ids"]                                          # [T]
    hit = (top[..., None, :] == ids[None, None, :, None]).any(-1)      # [L, P, T]
    hit &= (ids >= 0)[None, None, :]
    return hit & layer_mask(row, lens)


def partner(rows):
    """Map each prompt to another prompt of the same condition with a different problem."""
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    out = {}
    for group in by_cond.values():
        for i, r in enumerate(group):
            j = (i + 1) % len(group)
            while group[j]["problem_id"] == r["problem_id"]:
                j = (j + 1) % len(group)
            out[r["id"]] = group[j]
    return out


def summarize(rows, partners, lens, k, role, correct_only):
    """Per condition: real and chance P(target hit at any position), as [L, T] arrays."""
    real, chance, n = defaultdict(list), defaultdict(list), defaultdict(int)
    for r in rows:
        if (correct_only and not r["correct"]) or (role == "filler" and r["n_filler"] == 0):
            continue
        real[r["condition"]].append(real_hits(r, lens, k, role).any(axis=1))
        chance[r["condition"]].append(chance_hits(r, partners[r["id"]], lens, k, role).any(axis=1))
        n[r["condition"]] += 1
    return ({c: np.mean(v, axis=0) for c, v in real.items()},
            {c: np.mean(v, axis=0) for c, v in chance.items()}, dict(n))


def position_profile(rows, lens, k, target_index, correct_only=True):
    """For filler positions, counted from the END of the filler: P(hit) per (layer, pos_from_end),
    per condition, over the last 24 filler tokens (so sets with different lengths align)."""
    width = 24
    acc = defaultdict(list)
    for r in rows:
        if (correct_only and not r["correct"]) or r["n_filler"] < width:
            continue
        h = real_hits(r, lens, k, "filler")[:, -width:, target_index]   # [L, width]
        acc[r["condition"]].append(h)
    return {c: np.mean(v, axis=0) for c, v in acc.items()}


def raw_prob_profile(rows, lens, target_index, correct_only=True):
    """Mean over prompts of max over filler positions of the raw lens probability, per layer."""
    acc = defaultdict(list)
    for r in rows:
        if (correct_only and not r["correct"]) or r["n_filler"] == 0:
            continue
        p = np.nan_to_num(block(r[f"{lens}_rawp"], r, "filler")[..., target_index]) * layer_mask(r, lens)[:, :, 0]
        acc[r["condition"]].append(p.max(axis=1))
    return {c: np.mean(v, axis=0) for c, v in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", type=Path, default=Path("/root/readouts"))
    parser.add_argument("--out", type=Path, default=Path("/root/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.readouts)
    partners = partner(rows)
    targets = rows[0]["targets"]
    layers = rows[0]["layers"]
    print(f"{len(rows)} readouts; layers {layers[0]}..{layers[-1]}; targets {targets}")

    acc = {c: {"n": len(s), "accuracy": float(np.mean([r["correct"] for r in s])),
               "mean_answer_prob": float(np.nanmean([r["answer_prob"] for r in s]))}
           for c in CONDITIONS if (s := [r for r in rows if r["condition"] == c])}
    print("accuracy on these weights:", json.dumps(acc))

    summary = {"accuracy": acc, "targets": targets, "layers": layers, "rates": {}, "position_profiles": {}, "raw_prob": {}}
    for lens in LENSES:
        for role in ("filler", "last"):
            for k in KS:
                for correct_only in (True, False):
                    real, chance, n = summarize(rows, partners, lens, k, role, correct_only)
                    tag = f"{lens}/{role}/top{k}/{'correct' if correct_only else 'all'}"
                    summary["rates"][tag] = {
                        c: {"n": n[c],
                            "real_max": {t: float(real[c][:, i].max()) for i, t in enumerate(targets)},
                            "real_best_layer": {t: layers[int(real[c][:, i].argmax())] for i, t in enumerate(targets)},
                            "chance_at_best_layer": {t: float(chance[c][int(real[c][:, i].argmax()), i]) for i, t in enumerate(targets)},
                            "chance_max": {t: float(chance[c][:, i].max()) for i, t in enumerate(targets)},
                            "real_per_layer": {t: real[c][:, i].tolist() for i, t in enumerate(targets)},
                            "chance_per_layer": {t: chance[c][:, i].tolist() for i, t in enumerate(targets)}}
                        for c in real
                    }
        for t in ("y", "c2y", "answer"):
            ti = targets.index(t)
            prof = position_profile(rows, lens, 1, ti)
            summary["position_profiles"][f"{lens}/top1/{t}"] = {c: v.tolist() for c, v in prof.items()}
            rp = raw_prob_profile(rows, lens, ti)
            summary["raw_prob"][f"{lens}/{t}"] = {c: v.tolist() for c, v in rp.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    def show(tag, conds, cols):
        blk = summary["rates"][tag]
        for c in conds:
            if c not in blk:
                continue
            cells = []
            for t in cols:
                cells.append(f"{t}={blk[c]['real_max'][t]:.2f}(ch {blk[c]['chance_at_best_layer'][t]:.2f})@L{blk[c]['real_best_layer'][t]}")
            print(f"    {c:9s} n={blk[c]['n']:3d}  " + "  ".join(cells))

    computed = ["c1x", "y", "c2y", "answer"]
    print("\n== Residual TOP-1 at ANY filler position, correct answers only: real (chance at that layer) @ best layer")
    for lens in LENSES:
        print(f"  {lens}:")
        show(f"{lens}/filler/top1/correct", ("counting", "false", "true"), ["x"] + computed)
    print("\n== Residual TOP-10 at ANY filler position, correct answers only")
    for lens in LENSES:
        print(f"  {lens}:")
        show(f"{lens}/filler/top10/correct", ("counting", "false", "true"), ["x"] + computed)
    print("\n== Residual TOP-1 at the ANSWER position, correct answers only")
    for lens in LENSES:
        print(f"  {lens}:")
        show(f"{lens}/last/top1/correct", CONDITIONS, ["x"] + computed)
    print("\n== Same for INCORRECT answers only (filler positions, top-1)")
    for lens in ("logit",):
        real, chance, n = summarize([r for r in rows if not r["correct"]], partners, lens, 1, "filler", False)
        for c in ("counting", "false", "true"):
            if c in real:
                print(f"    {c:9s} n={n[c]:3d}  " + "  ".join(f"{t}={real[c][:, targets.index(t)].max():.2f}(ch {chance[c][:, targets.index(t)].max():.2f})" for t in ["x"] + computed))

    print("\n== WHERE in the filler (logit lens, top-1, correct only): P(hit) at the last filler token vs best position in the last 24")
    for t in ("y", "c2y", "answer"):
        prof = summary["position_profiles"][f"logit/top1/{t}"]
        for c in ("counting", "false", "true"):
            if c in prof:
                m = np.array(prof[c])  # [L, 24]
                li, pi = np.unravel_index(m.argmax(), m.shape)
                last = m[:, -1]
                print(f"    {t:6s} {c:9s} last token: {last.max():.2f}@L{layers[int(last.argmax())]} | best: {m.max():.2f}@L{layers[li]} pos_from_end={24 - 1 - pi}")

    print("\n== Raw lens probability of the ANSWER token at filler positions (mean over prompts of max over positions), best layer")
    for lens in LENSES:
        rp = summary["raw_prob"][f"{lens}/answer"]
        print(f"  {lens}: " + "  ".join(f"{c}={max(v):.3f}@L{layers[int(np.argmax(v))]}" for c, v in rp.items()))


if __name__ == "__main__":
    main()
