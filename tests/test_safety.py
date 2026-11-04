import pytest

from earshot.safety import is_supported, urgent_phrase


@pytest.mark.parametrize("text", ["Watch out!", "There's a FIRE", "call the police", "Gas leak here"])
def test_urgent_speech_is_found(text):
    assert urgent_phrase(text)


@pytest.mark.parametrize("text", ["The doctor will see you", "Your firewall is fine", "Fired up!"])
def test_ordinary_speech_is_not_urgent(text):
    assert urgent_phrase(text) is None


EVIDENCE = "Current medicines Insulin glargine 18 units every night at 10 pm"


def test_supported_reply_passes():
    assert is_supported("I take 18 units of glargine at 10 pm.", "What insulin do you take?", EVIDENCE)


def test_changed_dose_is_rejected():
    assert not is_supported("I take 28 units of glargine.", "What insulin do you take?", EVIDENCE)


def test_invented_fact_is_rejected():
    answer = "I take metformin and aspirin with breakfast."
    assert not is_supported(answer, "What do you take?", EVIDENCE)


def test_repeating_the_question_is_not_support():
    assert not is_supported("Yes, allergic.", "Are you allergic?", EVIDENCE)
