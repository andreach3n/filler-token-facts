"""Phase 3: aggregate lens readouts into per-condition decoding rates.

For each prompt, lens and layer we ask whether each target token (x, c1x, y, c2y, answer) is
among the top-k tokens of the cross-example residual at any filler position, and at the answer
position. Averaged over prompts per condition, that is paper 1's "decodable intermediate" rate.
Filler lengths differ slightly between statement sets, so position-resolved maps are kept per
(condition, set) and the headline numbers reduce over positions per prompt first.

  python aggregate.py --readouts /root/readouts --out /root/results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LENSES = ("logit", "jlens", "rlens")
CONDITIONS = ("none", "counting", "false", "true")
KS = (1, 5, 10)


def load(readouts: Path):
    rows = []
    for f in sorted(readouts.glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        rows.append({
            "id": meta["id"], "condition": meta["condition"], "set": meta["set"],
            "correct": int(d["greedy_token_id"]) == int(d["answer_token_id"]),
            "answer_prob": float(d["answer_prob"]),
            "n_filler": int(d["n_filler"]), "layers": d["layers"],
            "targets": list(d["targets"]),
            **{f"{lens}_rank": d[f"{lens}_target_res_rank"] for lens in LENSES},
        })
    return rows


def hits(row, lens, k, role):
    """[L, P, T] boolean: target in residual top-k, for the filler positions or the last one."""
    ranks = row[f"{lens}_rank"]
    block = ranks[:, : row["n_filler"]] if role == "filler" else ranks[:, -1:]
    return (block >= 0) & (block < k)


def rates(rows, lens, k, role, correct_only):
    """by_cond[c] = [L, T] P(target hit at any position of the role); by_set[(c, s)] = [L, P, T]."""
    any_pos, per_set = defaultdict(list), defaultdict(list)
    for r in rows:
        if correct_only and not r["correct"]:
            continue
        if role == "filler" and r["n_filler"] == 0:
            continue
        h = hits(r, lens, k, role)
        any_pos[r["condition"]].append(h.any(axis=1))
        per_set[(r["condition"], r["set"])].append(h)
    by_cond = {c: (np.mean(v, axis=0), len(v)) for c, v in any_pos.items()}
    by_set = {}
    for key, v in per_set.items():
        if len({x.shape for x in v}) == 1:
            by_set[key] = np.mean(v, axis=0)
    return by_cond, by_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", type=Path, default=Path("/root/readouts"))
    parser.add_argument("--out", type=Path, default=Path("/root/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.readouts)
    targets = rows[0]["targets"]
    layers = [int(l) for l in rows[0]["layers"]]
    print(f"{len(rows)} readouts; layers {layers[0]}..{layers[-1]}; targets {targets}")

    acc = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        if sub:
            acc[c] = {"n": len(sub), "accuracy": float(np.mean([r["correct"] for r in sub])),
                      "mean_answer_prob": float(np.nanmean([r["answer_prob"] for r in sub]))}
    print("accuracy on these weights:", json.dumps(acc))

    summary = {"accuracy": acc, "targets": targets, "layers": layers, "rates": {}}
    for lens in LENSES:
        for role in ("filler", "last"):
            for k in KS:
                for correct_only in (True, False):
                    by_cond, by_set = rates(rows, lens, k, role, correct_only)
                    tag = f"{lens}/{role}/top{k}/{'correct' if correct_only else 'all'}"
                    summary["rates"][tag] = {
                        c: {"n": n, "max_over_layers": {t: float(m[:, i].max()) for i, t in enumerate(targets)},
                            "best_layer": {t: layers[int(m[:, i].argmax())] for i, t in enumerate(targets)},
                            "per_layer": {t: m[:, i].tolist() for i, t in enumerate(targets)}}
                        for c, (m, n) in by_cond.items()
                    }
                    np.savez(args.out / f"posmap_{tag.replace('/', '_')}.npz",
                             **{f"{c}__{s}": v for (c, s), v in by_set.items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    def table(tag, conds):
        block = summary["rates"][tag]
        for c in conds:
            if c in block:
                m, b = block[c]["max_over_layers"], block[c]["best_layer"]
                print(f"    {c:9s} n={block[c]['n']:3d}  " + "  ".join(f"{t}={m[t]:.2f}@L{b[t]}" for t in targets))

    print("\nP(target in residual top-10 at ANY filler position), correct answers only; max over layers @ best layer")
    for lens in LENSES:
        print(f"  {lens}:")
        table(f"{lens}/filler/top10/correct", ("counting", "false", "true"))
    print("\nSame at the ANSWER position (last token), correct answers only")
    for lens in LENSES:
        print(f"  {lens}:")
        table(f"{lens}/last/top10/correct", CONDITIONS)
    print("\nTop-1 at any filler position, correct answers only")
    for lens in LENSES:
        print(f"  {lens}:")
        table(f"{lens}/filler/top1/correct", ("counting", "false", "true"))


if __name__ == "__main__":
    main()
