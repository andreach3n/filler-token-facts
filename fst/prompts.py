"""Prompt construction.

Layout follows paper 1: filler goes in the user turn between the question and "Answer:",
and every few-shot example carries the same filler as the target. Unlike paper 1, the system
prompt says nothing about the filler; paper 2 found that hint made no difference, and telling
the model that false statements are "space to think" would confound the result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fst.tasks import SYSTEM_PROMPTS, Problem

FILLER_PREFIX = "\n\nFiller: "
CORE_CONDITIONS = ("none", "counting", "false", "true")


def counting_filler(n: int) -> str:
    return " ".join(str(i) for i in range(1, n + 1))


def statement_filler(pairs: list[dict], truth: bool) -> str:
    return " ".join(pair["true_sentence" if truth else "false_sentence"] for pair in pairs)


def user_turn(problem: Problem, filler: str | None) -> str:
    if filler is None:
        return f"{problem.text}\n\nAnswer:"
    return f"{problem.text}{FILLER_PREFIX}{filler}\n\nAnswer:"


def build_prompt(problem: Problem, fewshot: list[Problem], filler: str | None) -> dict:
    """Provider-neutral chat prompt: a system string plus alternating user/assistant messages."""
    messages = []
    for example in fewshot:
        messages.append({"role": "user", "content": user_turn(example, filler)})
        messages.append({"role": "assistant", "content": str(example.answer)})
    messages.append({"role": "user", "content": user_turn(problem, filler)})
    return {"system": SYSTEM_PROMPTS[problem.task], "messages": messages}


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
