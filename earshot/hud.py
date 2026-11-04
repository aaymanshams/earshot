"""Fit text to a small glasses display.

This is plain code, not an LLM step, so the size limit can never be broken by a
model. Each page gets enough time on screen to be read comfortably.
"""
from .schemas import HudPage

WORDS_PER_MINUTE = 160
MIN_PAGE_SECONDS = 1.5


def _split_long_word(word: str, columns: int) -> list[str]:
    return [word[i : i + columns] for i in range(0, len(word), columns)]


def wrap(text: str, columns: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        for part in _split_long_word(word, columns):
            candidate = f"{line} {part}".strip()
            if len(candidate) <= columns:
                line = candidate
            else:
                lines.append(line)
                line = part
    if line:
        lines.append(line)
    return lines


def fit(text: str, columns: int = 32, rows: int = 2) -> list[HudPage]:
    lines = wrap(text, columns)
    pages = []
    for start in range(0, len(lines), rows):
        page = lines[start : start + rows]
        words = sum(len(line.split()) for line in page)
        seconds = max(MIN_PAGE_SECONDS, words / WORDS_PER_MINUTE * 60)
        pages.append(HudPage(lines=page, seconds=round(seconds, 2)))
    return pages
