"""Task generators.

Every problem stores its answer plus the intermediate values the interpretability pipeline
will look for at filler positions.

- system_of_equations: paper 1's chained variable-binding task (Brauer et al., scripts/data/
  generate_varbind_dataset.py), with every intermediate kept in 0..999 so each value is a
  single DeepSeek token.
- arithmetic: paper 2's multi-step arithmetic (Baherwani et al.): 5-7 nested operations,
  operands in [-99, 99]. Integer division and modulo only take a non-negative left operand
  and a positive right operand, so floor and truncation conventions agree.
- variable_counting: paper 2's code task: count the distinct variables assigned a value.
  Paper 2 gives no generator details, so the snippet design here is our own.
"""

from __future__ import annotations

import ast
import random
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path

SYSTEM_PROMPTS = {
    "system_of_equations": (
        "You will be given a list of variable definitions followed by a question. Each variable "
        "equals either a number or an expression that refers to an earlier variable (for example "
        "'twice the number for X plus 3'). Resolve the references to work out the value the "
        "question asks for, then answer immediately with just the number, nothing else. "
        "No explanation, no words, no reasoning, just the number."
    ),
    "arithmetic": (
        "You will be given a math problem. Answer immediately with just the number, nothing else. "
        "No explanation, no words, no reasoning, just the number."
    ),
    "variable_counting": (
        "You will be given a short Python code snippet followed by a question about it. Answer "
        "immediately with just the number, nothing else. No explanation, no words, no reasoning, "
        "just the number."
    ),
}

TASKS = tuple(SYSTEM_PROMPTS)


@dataclass
class Problem:
    task: str
    id: str
    text: str
    answer: int
    intermediates: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- system of equations -----------------------------------------------------------------

_COEF_WORDS = {2: "twice", 3: "three times"}
_VALUE_MAX = 999  # largest number DeepSeek's tokenizer keeps as one token


def _cvc_names() -> list[str]:
    """Consonant-vowel-consonant nonsense names, excluding real English words."""
    consonants = "bcdfghjklmnprstvwz"
    words: set[str] = set()
    dict_path = Path("/usr/share/dict/words")
    if dict_path.exists():
        words = {w.strip().lower() for w in dict_path.read_text().splitlines()}
    names = [c1 + v + c2 for c1 in consonants for v in "aeiou" for c2 in consonants]
    return [n for n in names if n not in words]


_NAMES = _cvc_names()


def _derived(rng: random.Random, base: int) -> tuple[int, str, int, int] | None:
    coef = rng.choice((2, 3))
    op = rng.choice(("plus", "minus"))
    const = rng.randint(1, 50)
    value = coef * base + const if op == "plus" else coef * base - const
    if not 0 <= value <= _VALUE_MAX:
        return None
    return coef, op, const, value


def system_of_equations(rng: random.Random, idx: int, num_terms: int = 5) -> Problem:
    while True:
        names = rng.sample(_NAMES, num_terms)
        x_name, y_name, *distractor_names = names
        x = rng.randint(10, 99)
        step1 = _derived(rng, x)
        step2 = _derived(rng, step1[3]) if step1 else None
        if step1 is None or step2 is None or step1[0] * x > _VALUE_MAX or step2[0] * step1[3] > _VALUE_MAX:
            continue
        c1, op1, k1, y = step1
        c2, op2, k2, answer = step2

        # Each line is (name, rendered value, names it depends on).
        lines = [(x_name, str(x), set()), (y_name, f"{_COEF_WORDS[c1]} the number for {x_name} {op1} {k1}", {x_name})]
        literals = {x_name: x}
        for name in distractor_names:
            spec = None
            if rng.random() < 0.5:
                ref = rng.choice(list(literals))
                spec = _derived(rng, literals[ref])
            if spec:
                coef, op, const, value = spec
                lines.append((name, f"{_COEF_WORDS[coef]} the number for {ref} {op} {const}", {ref}))
            else:
                value = rng.randint(10, 99)
                lines.append((name, str(value), set()))
            literals[name] = value

        # Random order that still defines every variable before it is referenced.
        ordered, placed, pending = [], set(), lines[:]
        while pending:
            ready = [line for line in pending if line[2] <= placed]
            line = rng.choice(ready)
            ordered.append(line)
            placed.add(line[0])
            pending.remove(line)

        definitions = "\n".join(f"{name} = {value}" for name, value, _ in ordered)
        question = f"What is {_COEF_WORDS[c2]} the number for {y_name} {op2} {k2}?"
        return Problem(
            task="system_of_equations",
            id=f"soe-{idx}",
            text=f"{definitions}\nQuestion: {question}",
            answer=answer,
            intermediates={"x": x, "c1x": c1 * x, "y": y, "c2y": c2 * y, "answer": answer},
        )


# --- multi-step arithmetic ---------------------------------------------------------------

_OPS = ("+", "-", "*", "//", "%")
_ARITH_BOUND = 9999


class _Retry(Exception):
    pass


def _build_expr(rng: random.Random, n_ops: int, values: list[int]) -> tuple[str, int]:
    if n_ops == 0:
        v = rng.randint(-99, 99)
        return (f"({v})" if v < 0 else str(v)), v
    n_left = rng.randint(0, n_ops - 1)
    left_s, left = _build_expr(rng, n_left, values)
    right_s, right = _build_expr(rng, n_ops - 1 - n_left, values)
    op = rng.choice(_OPS)
    if op in ("//", "%") and (left < 0 or right <= 0):
        raise _Retry
    if op == "+":
        value = left + right
    elif op == "-":
        value = left - right
    elif op == "*":
        value = left * right
    elif op == "//":
        value = left // right
    else:
        value = left % right
    if abs(value) > _ARITH_BOUND:
        raise _Retry
    values.append(value)
    return f"({left_s} {op} {right_s})", value


def arithmetic(rng: random.Random, idx: int) -> Problem:
    while True:
        values: list[int] = []
        try:
            expr, answer = _build_expr(rng, rng.randint(5, 7), values)
        except _Retry:
            continue
        expr = expr[1:-1]  # drop the outermost parentheses
        return Problem(
            task="arithmetic",
            id=f"arith-{idx}",
            text=f"Question: What is {expr}?",
            answer=answer,
            intermediates={"steps": values},
        )


# --- variable counting -------------------------------------------------------------------

_VAR_NAMES = [
    "count", "total", "index", "value", "result", "temp", "score", "offset", "limit", "size",
    "step", "width", "height", "level", "speed", "price", "rate", "delta", "flag", "items",
]


def count_assigned_variables(code: str) -> int:
    """Ground truth: distinct names that appear as an assignment target."""
    targets = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    targets.add(target.id)
    return len(targets)


def variable_counting(rng: random.Random, idx: int) -> Problem:
    n_distinct = rng.randint(3, 7)
    n_lines = rng.randint(max(8, n_distinct + 2), 12)
    names = rng.sample(_VAR_NAMES, n_distinct)
    # Positions of the lines that introduce each new variable; the first line always does.
    new_at = sorted(rng.sample(range(1, n_lines), n_distinct - 1))
    new_at = [0, *new_at]

    defined: list[str] = []
    lines, running = [], []
    for i in range(n_lines):
        if i in new_at:
            name = names[len(defined)]
            if defined and rng.random() < 0.5:
                rhs = f"{rng.choice(defined)} {rng.choice(('+', '-', '*'))} {rng.randint(1, 9)}"
            else:
                rhs = str(rng.randint(0, 99))
            lines.append(f"{name} = {rhs}")
            defined.append(name)
        else:
            kind = rng.choice(("augment", "reassign", "print"))
            name = rng.choice(defined)
            if kind == "augment":
                lines.append(f"{name} {rng.choice(('+=', '-=', '*='))} {rng.randint(1, 9)}")
            elif kind == "reassign":
                lines.append(f"{name} = {rng.choice(defined)} {rng.choice(('+', '-'))} {rng.randint(1, 9)}")
            else:
                lines.append(f"print({name})")
        running.append(len(defined))

    code = "\n".join(lines)
    assert count_assigned_variables(code) == n_distinct
    return Problem(
        task="variable_counting",
        id=f"varcount-{idx}",
        text=f"Code:\n{code}\nQuestion: How many distinct variables are assigned a value in this code?",
        answer=n_distinct,
        intermediates={"distinct_after_line": running},
    )


# --- splits ------------------------------------------------------------------------------

_GENERATORS = {"system_of_equations": system_of_equations, "arithmetic": arithmetic, "variable_counting": variable_counting}


def make_split(task: str, n_eval: int, n_fewshot: int, seed: int) -> tuple[list[Problem], list[Problem]]:
    """Generate disjoint few-shot and evaluation problems for one task."""
    rng = random.Random(f"{task}-{seed}")
    seen: set[str] = set()
    problems: list[Problem] = []
    while len(problems) < n_eval + n_fewshot:
        problem = _GENERATORS[task](rng, len(problems))
        if problem.text not in seen:
            seen.add(problem.text)
            problems.append(problem)
    return problems[:n_fewshot], problems[n_fewshot:]
