"""Behavioral pilot: core conditions x tasks x problems x models, via API.

Design: each problem is assigned to one statement set (problem i -> set i % n_sets) and run under
every core condition with that set's fillers, so conditions are compared on identical problems.
Few-shot examples are fixed per task and carry the same filler as the target.

  python -m fst.pilot build   # write prompts, print token and cost estimates (no API spend
                              #   except free Anthropic token counting for Opus filler lengths)
  python -m fst.pilot run     # send requests (costs money)
  python -m fst.pilot report  # accuracy, paired differences vs. no filler, format failures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from fst.models import MODELS, anthropic_count_after, parse_int_answer, run_requests
from fst.prompts import CORE_CONDITIONS, build_prompt, matched_fillers
from fst.statements import DATA_DIR as STATEMENTS_DIR
from fst.tasks import TASKS, make_split
from fst.tokenization import n_tokens, n_tokens_after

RUN_DIR = Path(__file__).resolve().parent.parent / "data" / "runs" / "pilot"
MAX_TOKENS = 16

# USD per million input/output tokens, looked up 2026-09-15 (Opus at Batch API rates). Re-check before big runs.
PRICES = {
    "opus-4.5": (2.50, 12.50),
    "deepseek-v4-flash": (0.081, 0.162),
    "deepseek-v3-0324": (0.25, 1.00),
    "qwen3.6-27b": (0.30, 2.00),
}
TASK_ABBR = {"system_of_equations": "soe", "arithmetic": "arith", "variable_counting": "varcount"}


def _count_after_for(model: str):
    spec = MODELS[model]
    if spec.tokenizer is None:
        return anthropic_count_after(spec.model_id)
    return lambda prefix, text: n_tokens_after(prefix, text, spec.tokenizer)


def build(models: list[str], tasks: list[str], n_problems: int, shots: int, seed: int) -> None:
    sets = json.loads((STATEMENTS_DIR / "sets.json").read_text())
    for model in models:
        count_after = _count_after_for(model)
        fillers = [matched_fillers(s["pairs"], count_after) for s in sets]
        requests, input_tokens = [], 0
        for task in tasks:
            fewshot, problems = make_split(task, n_problems, shots, seed)
            for i, problem in enumerate(problems):
                set_index = i % len(sets)
                for condition in CORE_CONDITIONS:
                    prompt = build_prompt(problem, fewshot, fillers[set_index].fillers[condition])
                    requests.append({
                        "id": f"{TASK_ABBR[task]}-{condition}-{i:03d}",
                        "task": task, "condition": condition, "set": sets[set_index]["id"],
                        "problem_id": problem.id, "answer": problem.answer, "prompt": prompt,
                    })
                    if MODELS[model].tokenizer:
                        text = prompt["system"] + "".join(m["content"] for m in prompt["messages"])
                        input_tokens += n_tokens(text, MODELS[model].tokenizer)

        out = RUN_DIR / "requests" / f"{model}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
        (RUN_DIR / "requests" / f"{model}.fillers.json").write_text(
            json.dumps([{"set": s["id"], **vars(f)} for s, f in zip(sets, fillers)], indent=1, ensure_ascii=False)
        )

        if not input_tokens:  # Claude's tokenizer is private; estimate from the DeepSeek tokenizer
            input_tokens = int(sum(
                n_tokens(r["prompt"]["system"] + "".join(m["content"] for m in r["prompt"]["messages"]), "deepseek-v4-flash")
                for r in requests
            ) * 1.1)
        price_in, price_out = PRICES[model]
        cost = input_tokens / 1e6 * price_in + len(requests) * MAX_TOKENS / 1e6 * price_out
        print(f"{model}: {len(requests)} requests, ~{input_tokens / 1e6:.2f}M input tokens, "
              f"worst-case cost ~${cost:.2f} -> {out}")
        for s, f in zip(sets, fillers):
            print(f"   {s['id']} filler tokens: {f.token_counts}")


def run(models: list[str]) -> None:
    for model in models:
        rows = [json.loads(line) for line in (RUN_DIR / "requests" / f"{model}.jsonl").read_text().splitlines()]
        run_requests(MODELS[model], {r["id"]: r["prompt"] for r in rows}, MAX_TOKENS, RUN_DIR / "responses")


def report(models: list[str]) -> None:
    for model in models:
        requests_path = RUN_DIR / "requests" / f"{model}.jsonl"
        responses_path = RUN_DIR / "responses" / f"{model}.jsonl"
        if not responses_path.exists():
            print(f"{model}: no responses yet")
            continue
        responses = {}
        for line in responses_path.read_text().splitlines():
            row = json.loads(line)
            responses[row["id"]] = row["result"]

        correct = defaultdict(dict)  # (task, condition) -> {problem index: bool}
        unparsed = defaultdict(int)
        hosts = defaultdict(int)
        for line in requests_path.read_text().splitlines():
            r = json.loads(line)
            result = responses.get(r["id"])
            if result is None or "error" in result:
                continue
            predicted = parse_int_answer(result["text"])
            if predicted is None:
                unparsed[(r["task"], r["condition"])] += 1
            hosts[result.get("host")] += 1
            correct[(r["task"], r["condition"])][r["id"].rsplit("-", 1)[1]] = predicted == r["answer"]

        print(f"\n== {model}  (hosts: {dict(hosts)})")
        for task in TASKS:
            base = correct.get((task, "none"))
            if not base:
                continue
            print(f"  {task}")
            for condition in CORE_CONDITIONS:
                cells = correct.get((task, condition), {})
                if not cells:
                    continue
                acc = sum(cells.values()) / len(cells)
                line = f"    {condition:9s} acc {acc:6.1%}  (n={len(cells)}, unparsed={unparsed[(task, condition)]})"
                if condition != "none":
                    shared = cells.keys() & base.keys()
                    gained = sum(cells[k] and not base[k] for k in shared)
                    lost = sum(base[k] and not cells[k] for k in shared)
                    p = binomtest(gained, gained + lost).pvalue if gained + lost else 1.0
                    delta = (gained - lost) / len(shared)
                    line += f"  vs none {delta:+6.1%}  (+{gained}/-{lost}, McNemar p={p:.3f})"
                print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "run", "report"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--n", type=int, default=150, help="problems per task")
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "build":
        build(args.models, args.tasks, args.n, args.shots, args.seed)
    elif args.command == "run":
        run(args.models)
    else:
        report(args.models)


if __name__ == "__main__":
    main()
