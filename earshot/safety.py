"""Rules that must work even when the model is slow, wrong or offline."""
import re

_URGENT = re.compile(
    r"\b(fire|watch out|look out|get out|evacuate|emergency|ambulance|call (the )?police"
    r"|gas leak|danger|bomb|stop right there|car coming|behind you)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.:]\d+)?")
_WORD = re.compile(r"[a-z0-9]+")
_IGNORED = frozenset(
    "a an and are am as at be but by can do for from have i if in is it me my no not of on or "
    "our so that the this to was we will with yes you your okay sure sorry think".split()
)


def urgent_phrase(text: str) -> str | None:
    match = _URGENT.search(text)
    return match.group(0) if match else None


def _stem(word: str) -> str:
    return word[:5]


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _IGNORED}


def is_supported(answer: str, question: str, evidence: str, min_coverage: float = 0.6) -> bool:
    """Check a drafted reply against the notes it cites, without trusting the model.

    Every number must appear in the evidence, because a wrong dose or time is the
    worst mistake this system could make. Most remaining words must appear too.
    Words already in the question do not count, so an answer cannot pass by
    repeating what was asked.
    """
    if not set(_NUMBER.findall(answer)) <= set(_NUMBER.findall(evidence)):
        return False
    claims = _words(answer) - _words(question)
    if not claims:
        return False
    return len(claims & _words(evidence)) / len(claims) >= min_coverage
