"""True/false statement pairs built from PopQA facts.

Pipeline:
  1. `candidates`: filter PopQA to well-known, single-answer facts with no numbers and write
     draft sentence pairs for review (data/statements/candidates.jsonl).
  2. Hand curation into data/statements/curated_spec.json: drop facts PopQA gets wrong, titles
     shared by several works, and co-credits; add a type word ("the film Alien"); define
     same-kind pools for false objects.
  3. `pairs`: for each curated fact, pick a false object from its pool with the same token
     length under every study tokenizer (data/statements/pairs.jsonl).
  4. Model verification (needs API keys): keep pairs every study model labels correctly.
  5. `sets`: assemble disjoint statement sets of 10 pairs, mixing relations.

Run: python -m fst.statements {candidates,pairs,sets}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from fst.tokenization import TOKENIZER_REPOS, n_tokens

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "statements"
PRIMARY_TOKENIZER = "deepseek-v4-flash"

# Two phrasings per relation so a set doesn't repeat one template.
TEMPLATES = {
    "author": ["{subj} was written by {obj}.", "{obj} is the author of {subj}."],
    "composer": ["The music for {subj} was composed by {obj}.", "{obj} composed the music for {subj}."],
    "director": ["{subj} was directed by {obj}.", "{obj} directed {subj}."],
    "sport": ["{subj} competes in {obj}.", "The sport played by {subj} is {obj}."],
}

MIN_SUBJECT_POP = 20_000  # monthly Wikipedia page views
MIN_OBJECT_POP = 10_000
_NUMBER_LIKE = re.compile(r"\d|\b[IVXLC]{2,}\b")


def _clean(text: str) -> bool:
    return not _NUMBER_LIKE.search(text)


def load_popqa():
    from datasets import load_dataset

    return load_dataset("akariasai/PopQA", split="test").to_pandas()


def build_candidates(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    df = load_popqa()
    df = df[df.prop.isin(TEMPLATES)]
    df["answers"] = df.possible_answers.map(lambda s: {a.lower() for a in json.loads(s)})

    # Keep subjects with exactly one row per relation whose object is among its listed answers;
    # PopQA lists several rows for multi-answer facts (e.g. every director of a TV series).
    rows_per_subject = df.groupby(["prop", "subj"]).subj.transform("size")
    df = df[(rows_per_subject == 1) & df.apply(lambda r: r.obj.lower() in r.answers, axis=1)]

    popular = df[(df.s_pop >= MIN_SUBJECT_POP) & (df.o_pop >= MIN_OBJECT_POP)]
    popular = popular[popular.subj.map(_clean) & popular.obj.map(_clean)]

    candidates = []
    for relation, group in popular.groupby("prop"):
        objects = sorted(set(df[(df.prop == relation) & (df.o_pop >= MIN_OBJECT_POP)].obj))
        objects = [o for o in objects if _clean(o)]
        for _, row in group.sort_values("s_pop", ascending=False).iterrows():
            template_id = rng.randrange(len(TEMPLATES[relation]))
            template = TEMPLATES[relation][template_id]
            true_sentence = template.format(subj=row.subj, obj=row.obj)
            target_len = n_tokens(true_sentence, PRIMARY_TOKENIZER)

            swaps = [o for o in objects if o.lower() not in row.answers and o != row.obj]
            rng.shuffle(swaps)
            false_obj = next(
                (o for o in swaps if n_tokens(template.format(subj=row.subj, obj=o), PRIMARY_TOKENIZER) == target_len),
                None,
            )
            if false_obj is None:
                continue
            candidates.append({
                "id": f"{relation.replace(' ', '_')}-{row.id}",
                "relation": relation,
                "subject": row.subj,
                "true_object": row.obj,
                "false_object": false_obj,
                "template_id": template_id,
                "true_sentence": true_sentence,
                "false_sentence": template.format(subj=row.subj, obj=false_obj),
                "n_tokens_deepseek": target_len,
                "subject_pop": int(row.s_pop),
                "object_pop": int(row.o_pop),
            })
    return candidates


def build_pairs(spec: dict, seed: int = 0) -> list[dict]:
    """Pick, for each curated entry, a same-kind false object with matching token length.

    Prefers a swap that matches under every study tokenizer; falls back to the primary one.
    Least-used pool members go first so no single false object dominates.
    """
    rng = random.Random(seed)
    usage = {name: 0 for pool in spec["pools"].values() for name in pool}
    pairs = []
    for i, entry in enumerate(spec["entries"]):
        template, true_object = entry["template"], entry["true_object"]
        true_sentence = template.format(obj=true_object)
        # Measured with a leading space: in the filler every sentence follows "Filler: " or ". ".
        true_lengths = {m: n_tokens(" " + true_sentence, m) for m in TOKENIZER_REPOS}

        options = [o for o in spec["pools"][entry["relation"]] if o.lower() != true_object.lower()]
        rng.shuffle(options)
        options.sort(key=lambda o: usage[o])
        chosen, matched = None, []
        for required in (list(TOKENIZER_REPOS), [PRIMARY_TOKENIZER]):
            chosen = next(
                (o for o in options if all(n_tokens(" " + template.format(obj=o), m) == true_lengths[m] for m in required)),
                None,
            )
            if chosen:
                matched = required
                break
        if chosen is None:
            print(f"  no token-matched false object for: {true_sentence}")
            continue
        usage[chosen] += 1
        pairs.append({
            "id": f"{entry['relation']}-{i:02d}",
            "relation": entry["relation"],
            "true_object": true_object,
            "false_object": chosen,
            "true_sentence": true_sentence,
            "false_sentence": template.format(obj=chosen),
            "n_tokens": true_lengths,
            "matched_tokenizers": matched,
        })
    return pairs


def build_sets(pairs: list[dict], per_relation: dict[str, int], n_sets: int, seed: int = 0) -> list[dict]:
    """Disjoint statement sets that mix relations and never name the same person or sport twice."""
    rng = random.Random(seed)
    by_relation: dict[str, list[dict]] = {}
    for pair in pairs:
        by_relation.setdefault(pair["relation"], []).append(pair)
    for group in by_relation.values():
        rng.shuffle(group)

    sets = []
    for s in range(n_sets):
        chosen, names = [], set()
        for relation, k in per_relation.items():
            for pair in list(by_relation[relation]):
                if len([p for p in chosen if p["relation"] == relation]) == k:
                    break
                if {pair["true_object"], pair["false_object"]} & names:
                    continue
                chosen.append(pair)
                names |= {pair["true_object"], pair["false_object"]}
                by_relation[relation].remove(pair)
        rng.shuffle(chosen)
        sets.append({"id": f"set-{s}", "pairs": chosen})
    return sets


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["candidates", "pairs", "sets"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "candidates":
        candidates = build_candidates(args.seed)
        _write_jsonl(DATA_DIR / "candidates.jsonl", candidates)
        print(f"wrote {len(candidates)} candidates to {DATA_DIR / 'candidates.jsonl'}")
    elif args.command == "pairs":
        spec = json.loads((DATA_DIR / "curated_spec.json").read_text())
        pairs = build_pairs(spec, args.seed)
        _write_jsonl(DATA_DIR / "pairs.jsonl", pairs)
        print(f"wrote {len(pairs)} of {len(spec['entries'])} pairs to {DATA_DIR / 'pairs.jsonl'}")
    else:
        source = DATA_DIR / "verified_pairs.jsonl"
        if not source.exists():
            print("verified_pairs.jsonl not found (run python -m fst.verify); building sets from unverified pairs")
            source = DATA_DIR / "pairs.jsonl"
        pairs = [json.loads(line) for line in source.read_text().splitlines()]
        sets = build_sets(pairs, per_relation={"author": 3, "composer": 3, "director": 2, "sport": 2}, n_sets=3, seed=args.seed)
        (DATA_DIR / "sets.json").write_text(json.dumps(sets, indent=1, ensure_ascii=False))
        print(f"wrote {len(sets)} sets of sizes {[len(s['pairs']) for s in sets]} to {DATA_DIR / 'sets.json'}")


if __name__ == "__main__":
    main()
