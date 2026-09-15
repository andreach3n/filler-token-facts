import re

import pytest

from fst.tasks import TASKS, count_assigned_variables, make_split

N = 300
_COEF = {"twice": 2, "three times": 3}
_DERIVED = re.compile(r"^(twice|three times) the number for (\w+) (plus|minus) (\d+)$")


def _apply(coef_word: str, value: int, op: str, const: str) -> int:
    base = _COEF[coef_word] * value
    return base + int(const) if op == "plus" else base - int(const)


def test_system_of_equations_answers_follow_from_the_text():
    _, problems = make_split("system_of_equations", N, 0, seed=0)
    for p in problems:
        definitions, question = p.text.split("\nQuestion: ")
        values = {}
        for line in definitions.splitlines():
            name, rhs = line.split(" = ")
            match = _DERIVED.match(rhs)
            if match:
                coef, ref, op, const = match.groups()
                assert ref in values, f"{ref} used before definition in {p.id}"
                values[name] = _apply(coef, values[ref], op, const)
            else:
                values[name] = int(rhs)
        coef, ref, op, const = re.match(r"What is (twice|three times) the number for (\w+) (plus|minus) (\d+)\?", question).groups()
        assert _apply(coef, values[ref], op, const) == p.answer
        assert all(0 <= v <= 999 for v in p.intermediates.values())


def test_arithmetic_answers_match_python():
    _, problems = make_split("arithmetic", N, 0, seed=0)
    for p in problems:
        expr = p.text.removeprefix("Question: What is ").removesuffix("?")
        assert eval(expr) == p.answer  # expressions contain only integers and operators
        assert 5 <= len(p.intermediates["steps"]) <= 7


def test_variable_counting_answers_match_ast_and_code_runs():
    _, problems = make_split("variable_counting", N, 0, seed=0)
    for p in problems:
        code = p.text.split("Code:\n")[1].split("\nQuestion:")[0]
        assert count_assigned_variables(code) == p.answer
        exec(compile(code, p.id, "exec"), {"print": lambda *a: None})
        assert 3 <= p.answer <= 7


@pytest.mark.parametrize("task", TASKS)
def test_splits_are_deterministic_and_disjoint(task):
    fewshot, problems = make_split(task, 50, 10, seed=3)
    again_fewshot, again = make_split(task, 50, 10, seed=3)
    assert [p.text for p in problems] == [p.text for p in again]
    assert [p.text for p in fewshot] == [p.text for p in again_fewshot]
    assert not {p.text for p in fewshot} & {p.text for p in problems}
