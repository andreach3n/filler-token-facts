"""Integrity checks on the generated statement data."""

import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data" / "statements"
PAIRS = [json.loads(line) for line in (DATA / "pairs.jsonl").read_text().splitlines()]
SETS = json.loads((DATA / "sets.json").read_text())


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p["id"])
def test_pair_differs_only_in_object(pair):
    assert pair["true_object"] != pair["false_object"]
    assert pair["true_sentence"].replace(pair["true_object"], "{obj}") == pair["false_sentence"].replace(pair["false_object"], "{obj}")
    assert not re.search(r"\d", pair["true_sentence"] + pair["false_sentence"])


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p["id"])
def test_pair_token_lengths_match(pair):
    from fst.tokenization import n_tokens

    for model in pair["matched_tokenizers"]:
        assert n_tokens(" " + pair["true_sentence"], model) == n_tokens(" " + pair["false_sentence"], model)


def test_sets_are_disjoint_and_never_repeat_a_name():
    seen_ids = set()
    for statement_set in SETS:
        ids = {p["id"] for p in statement_set["pairs"]}
        assert not ids & seen_ids
        seen_ids |= ids
        names = [n for p in statement_set["pairs"] for n in (p["true_object"], p["false_object"])]
        assert len(names) == len(set(names))
