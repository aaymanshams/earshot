import pytest

from earshot import hud


@pytest.mark.parametrize("text", ["short", "word " * 80, "x" * 200, "Take 2 tablets at 8:30 pm"])
def test_every_page_fits_the_display(text):
    for page in hud.fit(text, columns=32, rows=2):
        assert len(page.lines) <= 2
        assert all(len(line) <= 32 for line in page.lines)


def test_no_words_are_lost():
    text = "Please take this tablet twice a day after your food and drink some water"
    joined = " ".join(line for page in hud.fit(text) for line in page.lines)
    assert joined == text


def test_longer_pages_stay_on_screen_longer():
    short, long = hud.fit("Hi")[0], hud.fit("one two three four five six seven eight nine")[0]
    assert short.seconds >= hud.MIN_PAGE_SECONDS
    assert long.seconds >= short.seconds
