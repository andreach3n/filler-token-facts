"""Behavioral runs: conditions x tasks x problems x models, via API.

Design: each problem is assigned to one statement set (problem i -> set i % n_sets) and run under
every condition with that set's fillers, so conditions are compared on identical problems.
Conditions are no filler plus counting/false/true filler at each requested placement
(--placements after before middle). Few-shot examples are fixed per task and carry the same
filler, in the same place, as the target.

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
from fst.prompts import FILLER_KINDS, build_prompt, condition_name, matched_fillers, soe_chain_position
from fst.statements import DATA_DIR as STATEMENTS_DIR
from fst.tasks import TASKS, make_split
from fst.tokenization import n_tokens, n_tokens_after

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
MAX_TOKENS = 16

# USD per million input/output tokens, looked up 2026-09-15 (Opus at Batch API rates). Re-check before big runs.
PRICES = {
    "opus-4.5": (2.50, 12.50),
    "deepseek-v4-flash": (0.14, 0.28),  # Parasail; output price assumed 2x input, not looked up
    "deepseek-v3-0324": (0.29, 1.14),  # GMICloud
    "qwen3.6-27b": (0.30, 2.00),
}
TASK_ABBR = {"system_of_equations": "soe", "arithmetic": "arith", "variable_counting": "varcount"}


def _count_after_for(model: str):
    spec = MODELS[model]
    if spec.tokenizer is None:
        return anthropic_count_after(spec.model_id)
    return lambda prefix, text: n_tokens_after(prefix, text, spec.tokenizer)


def build(run_dir: Path, models: list[str], tasks: list[str], placements: list[str], n_problems: int, shots: int, seed: int) -> None:
    sets = json.loads((STATEMENTS_DIR / "sets.json").read_text())
    for model in models:
        count_after = _count_after_for(model)
        fillers = [matched_fillers(s["pairs"], count_after) for s in sets]
        requests, input_tokens = [], 0
        for task in tasks:
            fewshot, problems = make_split(task, n_problems, shots, seed)
            for i, problem in enumerate(problems):
                set_index = i % len(sets)
                cells = [("none", "after")] + [(kind, pl) for pl in placements for kind in FILLER_KINDS]
                for kind, placement in cells:
                    if placement == "middle" and task != "system_of_equations":
                        continue
                    condition = condition_name(kind, placement)
                    prompt = build_prompt(problem, fewshot, fillers[set_index].fillers[kind], placement)
                    requests.append({
                        "id": f"{TASK_ABBR[task]}-{condition}-{i:03d}",
                        "task": task, "condition": condition, "placement": placement, "set": sets[set_index]["id"],
                        "problem_id": problem.id, "answer": problem.answer, "prompt": prompt,
                    })
                    if MODELS[model].tokenizer:
                        text = prompt["system"] + "".join(m["content"] for m in prompt["messages"])
                        input_tokens += n_tokens(text, MODELS[model].tokenizer)

        out = run_dir / "requests" / f"{model}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
        (run_dir / "requests" / f"{model}.fillers.json").write_text(
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


def run(run_dir: Path, models: list[str]) -> None:
    for model in models:
        rows = [json.loads(line) for line in (run_dir / "requests" / f"{model}.jsonl").read_text().splitlines()]
        run_requests(MODELS[model], {r["id"]: r["prompt"] for r in rows}, MAX_TOKENS, run_dir / "responses")


def _mcnemar(cells: dict, base: dict) -> str:
    shared = cells.keys() & base.keys()
    gained = sum(cells[k] and not base[k] for k in shared)
    lost = sum(base[k] and not cells[k] for k in shared)
    p = binomtest(gained, gained + lost).pvalue if gained + lost else 1.0
    return f"vs none {(gained - lost) / len(shared):+6.1%}  (+{gained}/-{lost}, McNemar p={p:.3f})"


def report(run_dir: Path, models: list[str]) -> None:
    for model in models:
        requests_path = run_dir / "requests" / f"{model}.jsonl"
        responses_path = run_dir / "responses" / f"{model}.jsonl"
        if not responses_path.exists():
            print(f"{model}: no responses yet")
            continue
        responses = {}
        for line in responses_path.read_text().splitlines():
            row = json.loads(line)
            responses[row["id"]] = row["result"]

        correct = defaultdict(dict)  # (task, condition) -> {problem index: bool}
        order: dict[str, list[str]] = defaultdict(list)
        unparsed, hosts, missing = defaultdict(int), defaultdict(int), 0
        chain = {}  # problem index -> chain position (systems of equations)
        for line in requests_path.read_text().splitlines():
            r = json.loads(line)
            index = r["id"].rsplit("-", 1)[1]
            if r["condition"] not in order[r["task"]]:
                order[r["task"]].append(r["condition"])
            if r["task"] == "system_of_equations" and index not in chain:
                target = r["prompt"]["messages"][-1]["content"]
                problem_text = target.split("\n\nAnswer:")[0] if r["condition"] == "none" else None
                if problem_text:
                    chain[index] = soe_chain_position(problem_text)
            result = responses.get(r["id"])
            if result is None or "error" in result:
                missing += 1
                continue
            predicted = parse_int_answer(result["text"])
            if predicted is None:
                unparsed[(r["task"], r["condition"])] += 1
            hosts[result.get("host")] += 1
            correct[(r["task"], r["condition"])][index] = predicted == r["answer"]

        print(f"\n== {model}  (hosts: {dict(hosts)}, missing or errored: {missing})")
        for task in TASKS:
            base = correct.get((task, "none"))
            if not base:
                continue
            print(f"  {task}")
            for condition in order[task]:
                cells = correct.get((task, condition), {})
                if not cells:
                    continue
                acc = sum(cells.values()) / len(cells)
                line = f"    {condition:17s} acc {acc:6.1%}  (n={len(cells)}, unparsed={unparsed[(task, condition)]})"
                if condition != "none":
                    line += "  " + _mcnemar(cells, base)
                print(line)

            middle = [c for c in order[task] if c.endswith("-middle")]
            if middle and chain:
                print("    middle placement, split by where the queried chain sits relative to the filler:")
                for position in ("x,y before", "x before, y after", "x,y after"):
                    keys = {k for k, v in chain.items() if v == position}
                    sub_base = {k: v for k, v in base.items() if k in keys}
                    if not sub_base:
                        continue
                    print(f"      {position} (n={len(sub_base)}): none acc {sum(sub_base.values()) / len(sub_base):.1%}")
                    for condition in middle:
                        cells = {k: v for k, v in correct[(task, condition)].items() if k in keys}
                        print(f"        {condition:17s} acc {sum(cells.values()) / len(cells):6.1%}  " + _mcnemar(cells, sub_base))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "run", "report"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--placements", nargs="+", default=["after"], choices=["after", "before", "middle"])
    parser.add_argument("--n", type=int, default=150, help="problems per task")
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default="pilot", help="run name; each run keeps its own requests, responses and batch state")
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run
    if args.command == "build":
        build(run_dir, args.models, args.tasks, args.placements, args.n, args.shots, args.seed)
    elif args.command == "run":
        run(run_dir, args.models)
    else:
        report(run_dir, args.models)


if __name__ == "__main__":
    main()
