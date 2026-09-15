from fst.prompts import FILLER_PREFIX, build_prompt, counting_filler, matched_fillers, soe_chain_position, user_turn
from fst.tasks import SYSTEM_PROMPTS, make_split

PAIRS = [
    {"true_sentence": "The film Alien was directed by Ridley Scott.", "false_sentence": "The film Alien was directed by Woody Allen."},
    {"true_sentence": "Jason Kidd was a professional basketball player.", "false_sentence": "Jason Kidd was a professional cricket player."},
]


def _word_counter(prefix: str, text: str) -> int:
    return len((prefix + text).split()) - len(prefix.split())


def test_prompt_structure_and_filler_in_every_turn():
    fewshot, problems = make_split("system_of_equations", 1, 3, seed=0)
    prompt = build_prompt(problems[0], fewshot, filler="1 2 3")
    roles = [m["role"] for m in prompt["messages"]]
    assert roles == ["user", "assistant"] * 3 + ["user"]
    assert prompt["system"] == SYSTEM_PROMPTS["system_of_equations"]
    for message in prompt["messages"]:
        if message["role"] == "user":
            assert message["content"].endswith(f"{FILLER_PREFIX}1 2 3\n\nAnswer:")
        else:
            assert message["content"].lstrip("-").isdigit()


def test_no_filler_condition_has_no_filler_line():
    fewshot, problems = make_split("arithmetic", 1, 2, seed=0)
    prompt = build_prompt(problems[0], fewshot, filler=None)
    assert all("Filler:" not in m["content"] for m in prompt["messages"])


def test_matched_fillers_equalize_lengths():
    result = matched_fillers(PAIRS, _word_counter)
    assert result.fillers["none"] is None
    assert result.token_counts["false"] == result.token_counts["true"]
    assert abs(result.token_counts["counting"] - result.token_counts["false"]) <= 1
    assert result.fillers["counting"] == counting_filler(len(result.fillers["counting"].split()))


def test_placements_put_the_filler_where_expected():
    _, problems = make_split("system_of_equations", 20, 0, seed=0)
    for problem in problems:
        after = user_turn(problem, "F", "after")
        before = user_turn(problem, "F", "before")
        middle = user_turn(problem, "F", "middle")
        assert after == f"{problem.text}\n\nFiller: F\n\nAnswer:"  # unchanged from the pilot format
        assert before == f"Filler: F\n\n{problem.text}\n\nAnswer:"
        lines = problem.text.split("\n")
        assert middle.split("\n\nFiller: F\n\n") == ["\n".join(lines[:3]), "\n".join(lines[3:]) + "\n\nAnswer:"]


def test_middle_placement_rejects_other_tasks():
    import pytest

    _, problems = make_split("arithmetic", 1, 0, seed=0)
    with pytest.raises(ValueError):
        user_turn(problems[0], "F", "middle")


def test_chain_position_matches_definition_order():
    _, problems = make_split("system_of_equations", 200, 0, seed=0)
    seen = set()
    for problem in problems:
        position = soe_chain_position(problem.text)
        seen.add(position)
        names = [line.split(" = ")[0] for line in problem.text.split("\n")[:-1]]
        rhs = dict(line.split(" = ") for line in problem.text.split("\n")[:-1])
        y = problem.text.split("the number for ")[-1].split(" ")[0]
        x = rhs[y].split("the number for ")[1].split(" ")[0]
        expected = (names.index(x) < 3, names.index(y) < 3)
        assert position == {(True, True): "x,y before", (True, False): "x before, y after", (False, False): "x,y after"}[expected]
    assert seen == {"x,y before", "x before, y after", "x,y after"}
