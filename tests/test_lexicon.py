import pytest

from aikoru.lexicon import (
    OBJECT_NAMES,
    OBJECT_STEMS,
    PHRASES,
    distance_phrase,
    find_object,
    number_words,
    object_name,
    sentence,
    spell_digits,
)


def test_phrases_have_same_keys_for_both_languages():
    assert PHRASES["ru"].keys() == PHRASES["kk"].keys()


def test_every_stem_points_to_known_object():
    assert set(OBJECT_STEMS.values()) <= set(OBJECT_NAMES)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("найди дверь", "door"),
        ("нет двери", "door"),
        ("есікті тап", "door"),
        ("где лестница", "stairs"),
        ("жук көлігі", None),
        ("привет", None),
    ],
)
def test_find_object(text, expected):
    assert find_object(text) == expected


def test_find_object_multiword_stem():
    assert find_object("жүк көлігі қайда") == "truck"


def test_object_name_translation():
    assert object_name("door", "ru") == "дверь"
    assert object_name("door", "kk") == "есік"
    assert object_name("unicorn", "ru") is None


@pytest.mark.parametrize(
    "n, lang, expected",
    [
        (0, "ru", "ноль"),
        (5, "ru", "пять"),
        (21, "ru", "двадцать один"),
        (5, "kk", "бес"),
        (42, "kk", "қырық екі"),
        (150, "ru", "150"),
    ],
)
def test_number_words(n, lang, expected):
    assert number_words(n, lang) == expected


def test_spell_digits_replaces_all_numbers():
    assert spell_digits("через 3 метра 12 шагов", "ru") == "через три метра двенадцать шагов"


def test_distance_phrase():
    assert distance_phrase("close", None, "ru") == "близко"
    assert distance_phrase("close", 0.5, "ru") == PHRASES["ru"]["very_close"]
    assert distance_phrase("close", 1, "ru") == "около одного метра"
    assert distance_phrase("close", 3, "ru") == "около трёх метров"
    assert distance_phrase("close", 3, "kk") == "шамамен үш метр"
    assert distance_phrase("close", 50, "ru") == PHRASES["ru"]["far"]


def test_sentence_capitalizes_first_letter_only():
    assert sentence("дверь слева") == "Дверь слева"
    assert sentence("") == ""
