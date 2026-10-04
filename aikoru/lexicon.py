"""Словари объектов, фразы ответов и числительные для ru / kk."""
from __future__ import annotations

import re

LANGS = ("ru", "kk")

# en-название класса YOLOE -> (ru, kk) для озвучивания
OBJECT_NAMES = {
    "person": ("человек", "адам"),
    "car": ("машина", "көлік"),
    "bus": ("автобус", "автобус"),
    "truck": ("грузовик", "жүк көлігі"),
    "bicycle": ("велосипед", "велосипед"),
    "motorcycle": ("мотоцикл", "мотоцикл"),
    "door": ("дверь", "есік"),
    "stairs": ("лестница", "баспалдақ"),
    "traffic light": ("светофор", "бағдаршам"),
    "crosswalk": ("пешеходный переход", "жаяу жүргіншілер өткелі"),
    "pole": ("столб", "бағана"),
    "bench": ("скамейка", "ұзын орындық"),
    "chair": ("стул", "орындық"),
    "table": ("стол", "үстел"),
    "trash can": ("мусорный бак", "қоқыс жәшігі"),
    "dog": ("собака", "ит"),
    "cat": ("кошка", "мысық"),
    "bottle": ("бутылка", "бөтелке"),
    "cup": ("чашка", "кесе"),
    "backpack": ("рюкзак", "рюкзак"),
    "bag": ("сумка", "сөмке"),
    "laptop": ("ноутбук", "ноутбук"),
    "cell phone": ("телефон", "телефон"),
    "keys": ("ключи", "кілттер"),
    "window": ("окно", "терезе"),
    "tree": ("дерево", "ағаш"),
    "elevator": ("лифт", "лифт"),
    "wall": ("стена", "қабырға"),
    "glasses": ("очки", "көзілдірік"),
    "book": ("книга", "кітап"),
}

# Основы слов (ru и kk) -> en-класс. Нужны, чтобы понимать падежные формы:
# «найди дверь», «нет двери», «есікті тап».
OBJECT_STEMS = {
    # ru
    "человек": "person", "люд": "person", "пешеход": "person",
    "машин": "car", "автомобил": "car", "автобус": "bus", "грузовик": "truck",
    "велосипед": "bicycle", "мотоцикл": "motorcycle",
    "двер": "door", "лестниц": "stairs", "ступен": "stairs",
    "светофор": "traffic light", "переход": "crosswalk", "зебр": "crosswalk",
    "столб": "pole", "скамей": "bench", "скамь": "bench", "лавочк": "bench", "лавк": "bench",
    "стул": "chair", "стол": "table", "мусор": "trash can", "урн": "trash can",
    "собак": "dog", "кошк": "cat", "кот": "cat", "бутылк": "bottle",
    "чашк": "cup", "кружк": "cup", "рюкзак": "backpack", "сумк": "bag",
    "ноутбук": "laptop", "телефон": "cell phone", "ключ": "keys",
    "окн": "window", "окон": "window", "дерев": "tree", "лифт": "elevator",
    "стен": "wall", "очк": "glasses", "книг": "book",
    # kk
    "адам": "person", "көлік": "car", "жүк көлігі": "truck",
    "есік": "door", "баспалдақ": "stairs", "бағдаршам": "traffic light",
    "өткел": "crosswalk", "бағана": "pole", "орындық": "chair", "үстел": "table",
    "қоқыс": "trash can", "ит": "dog", "мысық": "cat", "бөтелке": "bottle",
    "кесе": "cup", "сөмке": "bag", "кілт": "keys", "терезе": "window",
    "ағаш": "tree", "қабырға": "wall", "көзілдірік": "glasses", "кітап": "book",
}

PHRASES = {
    "ru": {
        "loading": "Загружаю модели, подождите.",
        "ready": "Система готова. Скажите слово активации и команду.",
        "listening": "Слушаю.",
        "looking": "Смотрю.",
        "not_heard": "Команда не услышана.",
        "not_found": "Не вижу: {obj}.",
        "found": "{obj} {dir}, {dist}.",
        "danger": "Осторожно! {obj} {dir}, {dist}.",
        "appeared": "{dir}: {obj}, {dist}.",
        "new_person": "{dir}: незнакомый человек, {dist}.",
        "new_people": "Незнакомых людей: {n}. Ближайший {dir}, {dist}.",
        "known_person": "{dir}: {name}, {dist}.",
        "person_left": "{name} больше не видно.",
        "remembered": "Запомнила: {name}.",
        "no_face": "Лицо не найдено. Попросите человека повернуться к камере.",
        "need_name": "Скажите имя, например: запомни как Айгерим.",
        "who_none": "Людей рядом не вижу.",
        "unknown_person": "незнакомый человек",
        "forgot": "Удалено из памяти: {names}.",
        "forgot_none": "Такого человека нет в памяти.",
        "faces_off": "Распознавание лиц недоступно.",
        "paused": "Фоновые подсказки на паузе.",
        "resumed": "Фоновые подсказки включены.",
        "error": "Произошла ошибка.",
        "left": "слева", "center": "прямо", "right": "справа",
        "very_close": "совсем близко", "close": "близко", "far": "далеко",
        "approx_m": "около {n} {unit}",
    },
    "kk": {
        "loading": "Модельдер жүктелуде, күте тұрыңыз.",
        "ready": "Жүйе дайын. Белсендіру сөзін және команданы айтыңыз.",
        "listening": "Тыңдап тұрмын.",
        "looking": "Қарап жатырмын.",
        "not_heard": "Команда естілмеді.",
        "not_found": "{obj} көрінбейді.",
        "found": "{obj} {dir}, {dist}.",
        "danger": "Абайлаңыз! {obj} {dir}, {dist}.",
        "appeared": "{dir}: {obj}, {dist}.",
        "new_person": "{dir}: бейтаныс адам, {dist}.",
        "new_people": "Бейтаныс адамдар: {n}. Ең жақыны {dir}, {dist}.",
        "known_person": "{dir}: {name}, {dist}.",
        "person_left": "{name} енді көрінбейді.",
        "remembered": "Есте сақталды: {name}.",
        "no_face": "Бет табылмады. Адамнан камераға қарауын өтініңіз.",
        "need_name": "Есімін айтыңыз, мысалы: Айгерім деп есте сақта.",
        "who_none": "Жақын маңда адамдар көрінбейді.",
        "unknown_person": "бейтаныс адам",
        "forgot": "Жадтан өшірілді: {names}.",
        "forgot_none": "Мұндай адам жадта жоқ.",
        "faces_off": "Бет тану қолжетімсіз.",
        "paused": "Фондық кеңестер тоқтатылды.",
        "resumed": "Фондық кеңестер қосылды.",
        "error": "Қате орын алды.",
        "left": "сол жақта", "center": "алдыңызда", "right": "оң жақта",
        "very_close": "өте жақын", "close": "жақын", "far": "алыста",
        "approx_m": "шамамен {n} {unit}",
    },
}

# --- числительные -----------------------------------------------------------

_RU_UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь",
             "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
             "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_RU_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
            "семьдесят", "восемьдесят", "девяносто"]
_RU_GEN = ["", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми",
           "девяти", "десяти"]
_KK_UNITS = ["нөл", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз"]
_KK_TENS = ["", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен", "тоқсан"]


def number_words(n: int, lang: str) -> str:
    if not 0 <= n < 100:
        return str(n)
    if lang == "ru":
        if n < 20:
            return _RU_UNITS[n]
        t, u = divmod(n, 10)
        return _RU_TENS[t] + (f" {_RU_UNITS[u]}" if u else "")
    t, u = divmod(n, 10)
    if t == 0:
        return _KK_UNITS[u]
    return _KK_TENS[t] + (f" {_KK_UNITS[u]}" if u else "")


def spell_digits(text: str, lang: str) -> str:
    """TTS Silero не читает цифры — заменяем их словами."""
    return re.sub(r"\d+", lambda m: number_words(int(m.group()), lang), text)


def distance_phrase(distance: str, meters: float | None, lang: str) -> str:
    p = PHRASES[lang]
    if meters is None:
        return p[distance]
    if meters < 1:
        return p["very_close"]
    n = round(meters)
    if n > 10:
        return p["far"]
    if lang == "ru":
        return p["approx_m"].format(n=_RU_GEN[n], unit="метра" if n == 1 else "метров")
    return p["approx_m"].format(n=number_words(n, "kk"), unit="метр")


def sentence(text: str) -> str:
    """Первая буква предложения — заглавная."""
    return text[:1].upper() + text[1:]


def object_name(en: str, lang: str) -> str | None:
    pair = OBJECT_NAMES.get(en)
    return pair[LANGS.index(lang)] if pair else None


def find_object(text: str) -> str | None:
    """Ищет в тексте упоминание объекта из словаря, возвращает en-класс."""
    text = text.lower()
    for stem in sorted((s for s in OBJECT_STEMS if " " in s), key=len, reverse=True):
        if stem in text:
            return OBJECT_STEMS[stem]
    for word in re.findall(r"[\w-]+", text):
        for stem in sorted(OBJECT_STEMS, key=len, reverse=True):
            if " " in stem or not word.startswith(stem):
                continue
            # короткие основы («ит», «кот») — только с коротким окончанием
            if len(stem) >= 4 or len(word) - len(stem) <= 2:
                return OBJECT_STEMS[stem]
    return None
