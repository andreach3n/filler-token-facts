"""Keep only statement pairs that every study model classifies correctly.

A false statement only tests "computing over falsehoods" if the model knows it's false, so each
sentence of every pair is posed to every model as a true/false question. The run is cheap:
~84 sentences x 4 models with ~40-token prompts.

Run: python -m fst.verify [--models opus-4.5 deepseek-v4-flash ...]
Writes data/statements/verification/<model>.jsonl and data/statements/verified_pairs.jsonl.
"""

from __future__ import annotations

import argparse
import json

from fst.models import MODELS, run_requests
from fst.statements import DATA_DIR

SYSTEM = "Decide whether the statement is true or false. Reply with exactly one word: True or False."


def verification_requests(pairs: list[dict]) -> dict[str, dict]:
    requests = {}
    for pair in pairs:
        for truth in (True, False):
            sentence = pair["true_sentence" if truth else "false_sentence"]
            request_id = f"{pair['id']}_{'t' if truth else 'f'}"
            requests[request_id] = {"system": SYSTEM, "messages": [{"role": "user", "content": sentence}]}
    return requests


def label(text: str) -> bool | None:
    word = text.strip().strip(".").lower()
    return {"true": True, "false": False}.get(word.split()[0] if word else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    pairs = [json.loads(line) for line in (DATA_DIR / "pairs.jsonl").read_text().splitlines()]
    requests = verification_requests(pairs)
    passed = {pair["id"]: True for pair in pairs}

    for name in args.models:
        results = run_requests(MODELS[name], requests, max_tokens=5, out_dir=DATA_DIR / "verification")
        wrong = []
        for pair in pairs:
            for truth, suffix in ((True, "t"), (False, "f")):
                result = results.get(f"{pair['id']}_{suffix}", {})
                if label(result.get("text", "")) is not truth:
                    passed[pair["id"]] = False
                    wrong.append(f"{pair['id']}_{suffix}: {result.get('text', result.get('error'))!r}")
        print(f"{name}: {len(wrong)} of {len(requests)} sentences misclassified or unparsed")
        for line in wrong:
            print("   ", line)

    verified = [pair for pair in pairs if passed[pair["id"]]]
    (DATA_DIR / "verified_pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in verified))
    print(f"{len(verified)} of {len(pairs)} pairs verified by {', '.join(args.models)}")


if __name__ == "__main__":
    main()
