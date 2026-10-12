import pytest

from aikoru.voice import strip_wake, wake_score

WAKE = ["айкору"]


@pytest.mark.parametrize("heard", ["айкору", "ай кору", "а и кору", "әй көру", "ай көру что вокруг"])
def test_wake_word_variants_match(heard):
    assert wake_score(heard, WAKE) >= 0.75


@pytest.mark.parametrize("heard", ["и кому это нужно", "какой хороший день", "кто это"])
def test_ordinary_phrases_do_not_match(heard):
    assert wake_score(heard, WAKE) < 0.75


@pytest.mark.parametrize("heard,command", [
    ("айкору найди дверь", "найди дверь"),
    ("ай кору сколько людей", "сколько людей"),
    ("кору что вокруг", "что вокруг"),
    ("найди дверь", "найди дверь"),
])
def test_strip_wake(heard, command):
    assert strip_wake(heard, WAKE, 0.75) == command
