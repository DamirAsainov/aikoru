import pytest

from aikoru.commands import Command, parse, pick_language


class Hyp:
    def __init__(self, text, lang, conf):
        self.text, self.lang, self.conf = text, lang, conf


def test_parse_describe_command_ru():
    cmd = parse("что вокруг", "ru")
    assert cmd.intent == "describe"
    assert cmd.lang == "ru"


def test_parse_describe_command_kk():
    assert parse("айнала не бар", "kk").intent == "describe"


def test_parse_find_object_command_ru():
    cmd = parse("найди дверь", "ru")
    assert cmd.intent == "find"
    assert cmd.target_en == "door"


def test_parse_find_object_command_kk():
    cmd = parse("есікті тап", "kk")
    assert cmd.intent == "find"
    assert cmd.target_en == "door"


def test_parse_find_unknown_object_keeps_raw_phrase():
    cmd = parse("найди зонтик", "ru")
    assert cmd.intent == "find"
    assert cmd.target_en is None
    assert cmd.target_raw == "зонтик"


def test_parse_remember_face_ru():
    cmd = parse("запомни как Айгерим", "ru")
    assert cmd.intent == "remember"
    assert cmd.name == "Айгерим"


def test_parse_remember_face_kk():
    cmd = parse("Айгерім деп есте сақта", "kk")
    assert cmd.intent == "remember"
    assert cmd.name == "Айгерім"


def test_parse_forget_one_and_all():
    assert parse("забудь Айгерим", "ru").name == "Айгерим"
    assert parse("забудь всех", "ru").name == ""


def test_parse_who_command():
    assert parse("кто это", "ru").intent == "who"
    assert parse("бұл кім", "kk").intent == "who"


@pytest.mark.parametrize(
    "text, lang, intent",
    [
        ("стоп", "ru", "pause"),
        ("тоқта", "kk", "pause"),
        ("продолжай", "ru", "resume"),
        ("жалғастыр", "kk", "resume"),
    ],
)
def test_parse_pause_resume(text, lang, intent):
    assert parse(text, lang).intent == intent


def test_parse_free_question_and_unknown():
    assert parse("какого цвета небо сегодня", "ru").intent == "ask"
    assert parse("", "ru").intent == "unknown"
    assert parse("эээ", "ru").intent == "unknown"


def test_parse_is_case_insensitive():
    assert parse("Найди ДВЕРЬ", "ru").target_en == "door"


def test_pick_language_prefers_recognized_command():
    hyps = [Hyp("абракадабра", "ru", 0.4), Hyp("есікті тап", "kk", 0.4)]
    cmd = pick_language(hyps)
    assert isinstance(cmd, Command)
    assert cmd.intent == "find"
    assert cmd.lang == "kk"


def test_pick_language_empty():
    assert pick_language([]).intent == "unknown"
