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
ALL_WORDS = {"всех", "всё", "все", "барлығын", "бәрін", "барлық"}
# Слова-связки, которые не являются именем: «запомни этого человека как Айгерим»
NAME_FILLER = {
    "этого", "эту", "этот", "человека", "человек", "как", "его", "её", "ее", "это", "меня",
    "лицо", "зовут", "имя", "под", "именем", "пожалуйста", "он", "она", "друга", "подругу",
    "бұл", "мынаны", "мына", "адамды", "адам", "атымен", "есімі", "аты", "деп", "өтінемін", "бетін",
}


@dataclass
class Command:
    intent: str  # describe | find | who | remember | forget | pause | resume | ask | unknown
    lang: str
    text: str
    target_en: str | None = None   # для find: класс из словаря
    target_raw: str = ""           # для find: исходная фраза объекта, если нет в словаре
    name: str = ""                 # для remember/forget: имя человека


def _has(text: str, keys: list[str]) -> bool:
    return any(re.search(rf"(^|\s){re.escape(k)}", text) for k in keys)


def _strip_keys(text: str, keys: list[str]) -> str:
    for k in sorted(keys, key=len, reverse=True):
        text = re.sub(rf"(^|\s){re.escape(k)}\w*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_name(text: str, keys: list[str]) -> str:
    words = [w for w in re.findall(r"[\w-]+", _strip_keys(text, keys)) if w not in NAME_FILLER]
    return " ".join(w.capitalize() for w in words[:3])


def parse(text: str, lang: str) -> Command:
    t = text.lower().strip()
    if not t:
        return Command("unknown", lang, t)
    if _has(t, PAUSE[lang]):
        return Command("pause", lang, t)
    if _has(t, RESUME[lang]):
        return Command("resume", lang, t)
    if _has(t, REMEMBER[lang]):
        return Command("remember", lang, t, name=_extract_name(t, REMEMBER[lang]))
    if _has(t, FORGET[lang]):
        name = _extract_name(t, FORGET[lang])
        return Command("forget", lang, t, name="" if name.lower() in ALL_WORDS else name)
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
        if cmd.intent in ("describe", "find", "pause", "resume", "remember", "forget", "who"):
            score += 0.5
        if h.lang == "kk" and KK_LETTERS & set(h.text):
            score += 0.2
        if not h.text:
            score = -1
        scored.append((score, cmd))
    if not scored:
        return Command("unknown", "ru", "")
    return max(scored, key=lambda x: x[0])[1]
