"""Prompt construction.

Layout follows paper 1: by default filler goes in the user turn between the question and
"Answer:", and every few-shot example carries the same filler (in the same place) as the target.
Unlike paper 1, the system prompt says nothing about the filler; paper 2 found that hint made no
difference, and telling the model that false statements are "space to think" would confound
the result.

Placements (controls for where the filler has to sit to help):
- after:  problem, question, filler, Answer:  (main condition)
- before: filler, problem, question, Answer:  (paper 1 found no gain here)
- middle: systems of equations only; filler after the 3rd of 5 definitions, so some problems have
          the relevant chain (x, then y) defined before the filler and some don't
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from fst.tasks import SYSTEM_PROMPTS, Problem

FILLER_PREFIX = "\n\nFiller: "
FILLER_KINDS = ("counting", "false", "true")
PLACEMENTS = ("after", "before", "middle")
MIDDLE_AFTER_LINE = 3


def counting_filler(n: int) -> str:
    return " ".join(str(i) for i in range(1, n + 1))


def statement_filler(pairs: list[dict], truth: bool) -> str:
    return " ".join(pair["true_sentence" if truth else "false_sentence"] for pair in pairs)


def condition_name(kind: str, placement: str) -> str:
    """'none', or the filler kind with a placement suffix unless it's the default 'after'."""
    return kind if kind == "none" or placement == "after" else f"{kind}-{placement}"


def user_turn(problem: Problem, filler: str | None, placement: str = "after") -> str:
    if filler is None:
        return f"{problem.text}\n\nAnswer:"
    if placement == "after":
        return f"{problem.text}{FILLER_PREFIX}{filler}\n\nAnswer:"
    if placement == "before":
        return f"Filler: {filler}\n\n{problem.text}\n\nAnswer:"
    if placement == "middle":
        if problem.task != "system_of_equations":
            raise ValueError("middle placement is only defined for systems of equations")
        lines = problem.text.split("\n")
        head, tail = "\n".join(lines[:MIDDLE_AFTER_LINE]), "\n".join(lines[MIDDLE_AFTER_LINE:])
        return f"{head}{FILLER_PREFIX}{filler}\n\n{tail}\n\nAnswer:"
    raise ValueError(f"unknown placement {placement!r}")


def build_prompt(problem: Problem, fewshot: list[Problem], filler: str | None, placement: str = "after") -> dict:
    """Provider-neutral chat prompt: a system string plus alternating user/assistant messages."""
    messages = []
    for example in fewshot:
        messages.append({"role": "user", "content": user_turn(example, filler, placement)})
        messages.append({"role": "assistant", "content": str(example.answer)})
    messages.append({"role": "user", "content": user_turn(problem, filler, placement)})
    return {"system": SYSTEM_PROMPTS[problem.task], "messages": messages}


def build_prompt_per_shot(problem: Problem, fewshot: list[Problem], shot_fillers: list[str | None],
                          target_filler: str | None, placement: str = "after") -> dict:
    """Like build_prompt, but each few-shot example carries its own filler (e.g. a different
    statement set per shot, so the target's statements appear only once in the context)."""
    assert len(shot_fillers) == len(fewshot)
    messages = []
    for example, filler in zip(fewshot, shot_fillers):
        messages.append({"role": "user", "content": user_turn(example, filler, placement)})
        messages.append({"role": "assistant", "content": str(example.answer)})
    messages.append({"role": "user", "content": user_turn(problem, target_filler, placement)})
    return {"system": SYSTEM_PROMPTS[problem.task], "messages": messages}


def soe_chain_position(problem_text: str) -> str:
    """For middle placement: where the queried chain sits relative to the filler.

    Returns 'x,y before' (the model could compute y at filler positions), 'x before, y after',
    or 'x,y after'. y always follows x, since y's definition refers to x.
    """
    lines = problem_text.split("\n")
    y_name = re.search(r"the number for (\w+) (?:plus|minus) \d+\?$", lines[-1]).group(1)
    y_line = next(i for i, line in enumerate(lines) if line.startswith(f"{y_name} = "))
    x_name = re.search(r"the number for (\w+)", lines[y_line]).group(1)
    x_line = next(i for i, line in enumerate(lines) if line.startswith(f"{x_name} = "))
    x_before, y_before = x_line < MIDDLE_AFTER_LINE, y_line < MIDDLE_AFTER_LINE
    return "x,y before" if y_before else ("x before, y after" if x_before else "x,y after")


@dataclass
class ConditionFillers:
    """The filler string for each condition, token-matched for one model."""

    fillers: dict[str, str | None]
    token_counts: dict[str, int]


def matched_fillers(pairs: list[dict], count_after: Callable[[str, str], int]) -> ConditionFillers:
    """Fillers for the core conditions, with counting length chosen to match the statement block.

    `count_after(prefix, text)` returns the tokens `text` adds after `prefix` for the target model.
    True and false sentences are built to have equal token counts, so the statements set the target.
    """
    false_text = statement_filler(pairs, truth=False)
    true_text = statement_filler(pairs, truth=True)
    target = count_after(FILLER_PREFIX, false_text)

    # Counting-filler length grows with n, so binary-search the smallest n that reaches the target.
    # Keeps count_after calls to ~10, which matters when counts come from an API.
    def length(n: int) -> int:
        return count_after(FILLER_PREFIX, counting_filler(n))

    lo, hi = 1, 400
    while lo < hi:
        mid = (lo + hi) // 2
        if length(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    best_n = min((n for n in (lo - 1, lo) if n >= 1), key=lambda n: abs(length(n) - target))
    fillers = {"none": None, "counting": counting_filler(best_n), "false": false_text, "true": true_text}
    counts = {name: (0 if text is None else count_after(FILLER_PREFIX, text)) for name, text in fillers.items()}
    return ConditionFillers(fillers=fillers, token_counts=counts)
