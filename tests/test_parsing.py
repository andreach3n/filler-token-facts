import pytest

from fst.models import parse_int_answer
from fst.verify import label


@pytest.mark.parametrize(
    "text, expected",
    [("42", 42), (" -17\n", -17), ("Answer: 316", 316), ("1,234", 1234), ("The answer", None), ("", None)],
)
def test_parse_int_answer(text, expected):
    assert parse_int_answer(text) == expected


@pytest.mark.parametrize("text, expected", [("True", True), ("false.", False), (" TRUE ", True), ("Maybe", None), ("", None)])
def test_verification_label(text, expected):
    assert label(text) is expected
