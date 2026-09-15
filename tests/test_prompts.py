from fst.prompts import FILLER_PREFIX, build_prompt, counting_filler, matched_fillers
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
