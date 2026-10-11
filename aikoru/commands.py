"""Разбор голосовых команд на ru / kk."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import find_object

KK_LETTERS = set("әғқңөұүһі")

DESCRIBE = {
    "ru": ["что вокруг", "что впереди", "что передо мной", "опиши", "что видишь",
           "обстановк", "осмотрись", "где я"],
    "kk": ["айнала", "не бар", "сипатта", "не көріп", "алда не", "не көрінеді", "қайдамын"],
}
FIND = {
    "ru": ["найди", "найти", "где", "поищи", "ищи", "есть ли", "покажи"],
    "kk": ["тап", "қайда", "ізде", "бар ма"],
}
PAUSE = {"ru": ["стоп", "пауза", "тише", "замолчи", "хватит"], "kk": ["тоқта", "үндеме", "жетер"]}
RESUME = {"ru": ["продолж", "включи подсказки", "возобнов"], "kk": ["жалғастыр", "кеңестерді қос"]}
REMEMBER = {"ru": ["запомни"], "kk": ["есте сақта", "есіңе сақта", "сақтап ал", "сақта"]}
FORGET = {"ru": ["забудь"], "kk": ["ұмыт"]}
WHO = {
    "ru": ["кто это", "кто рядом", "кто передо мной", "кто там", "кто здесь", "кого видишь"],
    "kk": ["бұл кім", "кім бар", "кім тұр", "кім көрінеді", "кімді көріп"],
}
MEMORY = {
    "ru": ["что ты помнишь", "что помнишь", "кого ты помнишь", "кого помнишь", "что в памяти"],
    "kk": ["не есіңде", "кім есіңде", "жадта не бар"],
}
ALL_WORDS = {"всех", "всё", "все", "барлығын", "бәрін", "барлық"}
# Подсказки, кого запоминать: предмет или человека. Без них решает наличие лица в кадре.
OBJECT_WORDS = {"предмет", "предмета", "вещь", "вещи", "объект", "штуку", "штука",
                "зат", "затты", "затымды", "нәрсе", "нәрсені"}
POSSESSIVE = {"мой", "моя", "мою", "мои", "моё", "мое", "свой", "свою", "свои", "менің", "өзімнің"}
PERSON_WORDS = {"человека", "человек", "лицо", "зовут", "меня", "его", "её", "ее", "друга", "подругу",
                "он", "она", "адамды", "адам", "бетін", "есімі", "аты"}
# Слова-связки, которые не являются именем: «запомни этого человека как Айгерим»
NAME_FILLER = {
    "этого", "эту", "этот", "человека", "человек", "как", "его", "её", "ее", "это", "меня",
    "лицо", "зовут", "имя", "под", "именем", "пожалуйста", "он", "она", "друга", "подругу",
    *OBJECT_WORDS, *POSSESSIVE,
    "бұл", "мынаны", "мына", "адамды", "адам", "атымен", "есімі", "аты", "деп", "өтінемін", "бетін",
}


@dataclass
class Command:
    intent: str  # describe | find | who | remember | forget | memory | pause | resume | ask | unknown
    lang: str
    text: str
    target_en: str | None = None   # для find: класс из словаря
    target_raw: str = ""           # для find: исходная фраза объекта, если нет в словаре
    name: str = ""                 # для remember/forget: имя человека или предмета
    obj: bool | None = None        # для remember/forget: True — предмет, False — человек, None — неясно


def _has(text: str, keys: list[str]) -> bool:
    return any(re.search(rf"(^|\s){re.escape(k)}", text) for k in keys)


def _strip_keys(text: str, keys: list[str]) -> str:
    for k in sorted(keys, key=len, reverse=True):
        text = re.sub(rf"(^|\s){re.escape(k)}\w*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_name(text: str, keys: list[str]) -> str:
    words = [w for w in re.findall(r"[\w-]+", _strip_keys(text, keys)) if w not in NAME_FILLER]
    return " ".join(w.capitalize() for w in words[:3])


def _object_hint(text: str, name: str) -> bool | None:
    words = set(re.findall(r"[\w-]+", text))
    if words & PERSON_WORDS:
        return False
    if words & (OBJECT_WORDS | POSSESSIVE) or (name and find_object(name)):
        return True
    return None


def _memory_command(intent: str, t: str, lang: str, keys: list[str]) -> Command:
    name = _extract_name(t, keys)
    obj = _object_hint(t, name)
    if obj:
        name = name.lower()  # «ключи», а не «Ключи»: имя предмета звучит внутри фразы
    if intent == "forget" and name.lower() in ALL_WORDS:
        name = ""
    return Command(intent, lang, t, name=name, obj=obj)


def parse(text: str, lang: str) -> Command:
    t = text.lower().strip()
    if not t:
        return Command("unknown", lang, t)
    if _has(t, PAUSE[lang]):
        return Command("pause", lang, t)
    if _has(t, RESUME[lang]):
        return Command("resume", lang, t)
    if _has(t, MEMORY[lang]):
        return Command("memory", lang, t)
    if _has(t, REMEMBER[lang]):
        return _memory_command("remember", t, lang, REMEMBER[lang])
    if _has(t, FORGET[lang]):
        return _memory_command("forget", t, lang, FORGET[lang])
    if _has(t, WHO[lang]):
        return Command("who", lang, t)
    target = find_object(t)
    wants_find = _has(t, FIND[lang])
    if wants_find and target:
        return Command("find", lang, t, target, _strip_keys(t, FIND[lang]))
    if _has(t, DESCRIBE[lang]):
        return Command("describe", lang, t)
    if wants_find:
        raw = _strip_keys(t, FIND[lang])
        if raw:
            return Command("find", lang, t, None, raw)
    if target and len(t.split()) <= 2:
        return Command("find", lang, t, target, t)
    if len(t.split()) >= 2:
        return Command("ask", lang, t)
    return Command("unknown", lang, t)


def pick_language(hyps) -> Command:
    """Из гипотез разных моделей Vosk выбирает самую правдоподобную команду."""
    scored = []
    for h in hyps:
        cmd = parse(h.text, h.lang)
        score = h.conf
        if cmd.intent in ("describe", "find", "pause", "resume", "remember", "forget", "who",
                          "memory"):
            score += 0.5
        if h.lang == "kk" and KK_LETTERS & set(h.text):
            score += 0.2
        if not h.text:
            score = -1
        scored.append((score, cmd))
    if not scored:
        return Command("unknown", "ru", "")
    return max(scored, key=lambda x: x[0])[1]
