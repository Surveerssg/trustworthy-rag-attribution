"""LLM-as-attributor: reading the judge's passage choice from its first-token log-probs, with a text fallback."""
import math

from scripts.llm_attribute import parse


def reply(text, first=None, alts=()):
    r = {"message": {"content": text}}
    if first is not None:
        r["logprobs"] = [{"token": first[0], "logprob": first[1], "top_logprobs": [{"token": t, "logprob": l} for t, l in alts]}]
    return r


def test_probabilities_from_first_token():
    probs, pick, ok = parse(reply("3", ("3", math.log(0.7)), [("3", math.log(0.7)), ("1", math.log(0.2)), (" 5", math.log(0.05))]), 5)
    assert ok and pick == 3
    assert abs(probs[2] - 0.7) < 1e-9 and abs(probs[0] - 0.2) < 1e-9 and abs(probs[4] - 0.05) < 1e-9


def test_out_of_range_digits_are_ignored_and_text_fallback_works():
    probs, pick, ok = parse(reply("Passage 2", ("Pass", -0.1), [("Pass", -0.1), ("7", -2.0)]), 5)
    assert ok and pick == 2 and probs == [0, 1.0, 0, 0, 0]


def test_unicode_digit_alternatives_are_ignored():
    # Qwen offered '₂' (subscript two) among the top alternatives; str.isdigit() accepts it but int() does not
    probs, pick, ok = parse(reply("2", ("2", math.log(0.9)), [("2", math.log(0.9)), ("₂", math.log(0.05)), ("²", -5.0)]), 5)
    assert ok and pick == 2 and abs(sum(probs) - 0.9) < 1e-9


def test_no_passage_number_is_a_miss():
    assert parse(reply("None of them"), 5) == ([0.0] * 5, None, False)
