"""Phase 3: aggregate lens readouts into per-condition decoding rates.

For each (condition, lens, layer, position) we ask: across prompts, how often is each target
token (x, c1x, y, c2y, answer) among the top-k tokens of the cross-example residual? That is
paper 1's "decodable intermediate" measure. We report it for filler positions (the statement /
counting tokens) and for the final positions (where the answer is produced), split by whether
the model answered correctly.

  python aggregate.py --readouts /workspace/readouts --out results/gpu
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
            "targets": list(d["targets"]), "target_ids": d["target_ids"],
            **{k: d[k] for k in d.files if any(k.startswith(l + "_") for l in LENSES)},
        })
    return rows


def decoding_rates(rows, lens, k, role, correct_only):
    """Fraction of prompts with each target in the residual top-k, per (layer, position).

    role: 'filler' (positions 0..n_filler-1) or 'last' (the final 4 positions).
    Returns {condition: array [L, P, T]} averaged over prompts; P differs by role/condition.
    """
    per_cond = defaultdict(list)
    for r in rows:
        if correct_only and not r["correct"]:
            continue
        ranks = r[f"{lens}_target_res_rank"]  # [L, P, T], -1 where the target is not a single token
        if role == "filler":
            if r["n_filler"] == 0:
                continue
            block = ranks[:, : r["n_filler"]]
        else:
            block = ranks[:, -4:]
        hit = (block >= 0) & (block < k)
        per_cond[r["condition"]].append(hit.astype(np.float32))
    return {c: np.mean(v, axis=0) for c, v in per_cond.items() if v}, {c: len(v) for c, v in per_cond.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", type=Path, default=Path("/root/readouts"))
    parser.add_argument("--out", type=Path, default=Path("results/gpu"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.readouts)
    targets = rows[0]["targets"]
    layers = rows[0]["layers"]
    print(f"{len(rows)} readouts")

    # Accuracy on the exact weights we analyse, per condition.
    acc = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        if sub:
            acc[c] = {"n": len(sub), "accuracy": float(np.mean([r["correct"] for r in sub])),
                      "mean_answer_prob": float(np.nanmean([r["answer_prob"] for r in sub]))}
    print("accuracy on the pod:", json.dumps(acc, indent=1))

    summary = {"accuracy": acc, "targets": targets, "layers": layers.tolist(), "rates": {}}
    for lens in LENSES:
        for role in ("filler", "last"):
            for k in KS:
                for correct_only in (True, False):
                    rates, counts = decoding_rates(rows, lens, k, role, correct_only)
                    key = f"{lens}/{role}/top{k}/{'correct' if correct_only else 'all'}"
                    summary["rates"][key] = {
                        c: {"n": counts[c],
                            # best layer/position per target: the headline "is it decodable anywhere?"
                            "max_over_layer_pos": {t: float(rates[c][:, :, i].max()) for i, t in enumerate(targets)},
                            # per-layer max over positions, for layer curves
                            "per_layer_max": {t: rates[c][:, :, i].max(axis=1).tolist() for i, t in enumerate(targets)}}
                        for c in rates
                    }
                    np.savez(args.out / f"rates_{lens}_{role}_top{k}_{'correct' if correct_only else 'all'}.npz", **rates)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    # Headline table: residual top-10 among correct answers, best (layer, position), by condition.
    print("\nMax over (layer, position) of P(target in residual top-10), correct answers only")
    for lens in LENSES:
        block = summary["rates"][f"{lens}/filler/top10/correct"]
        print(f"  {lens} at FILLER positions")
        for c in ("counting", "false", "true"):
            if c in block:
                m = block[c]["max_over_layer_pos"]
                print(f"    {c:9s} n={block[c]['n']:3d} " + "  ".join(f"{t}={m[t]:.2f}" for t in targets))
        block = summary["rates"][f"{lens}/last/top10/correct"]
        print(f"  {lens} at the LAST 4 positions")
        for c in CONDITIONS:
            if c in block:
                m = block[c]["max_over_layer_pos"]
                print(f"    {c:9s} n={block[c]['n']:3d} " + "  ".join(f"{t}={m[t]:.2f}" for t in targets))


if __name__ == "__main__":
    main()
