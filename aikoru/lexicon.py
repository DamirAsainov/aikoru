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

# Формы для счёта по-русски: (1, 2–4, 5+, род). Род нужен для «одна/две», «одно».
RU_COUNT_FORMS = {
    "person": ("человек", "человека", "человек", "m"),
    "car": ("машина", "машины", "машин", "f"),
    "bus": ("автобус", "автобуса", "автобусов", "m"),
    "truck": ("грузовик", "грузовика", "грузовиков", "m"),
    "bicycle": ("велосипед", "велосипеда", "велосипедов", "m"),
    "motorcycle": ("мотоцикл", "мотоцикла", "мотоциклов", "m"),
    "door": ("дверь", "двери", "дверей", "f"),
    "stairs": ("лестница", "лестницы", "лестниц", "f"),
    "traffic light": ("светофор", "светофора", "светофоров", "m"),
    "crosswalk": ("пешеходный переход", "пешеходных перехода", "пешеходных переходов", "m"),
    "pole": ("столб", "столба", "столбов", "m"),
    "bench": ("скамейка", "скамейки", "скамеек", "f"),
    "chair": ("стул", "стула", "стульев", "m"),
    "table": ("стол", "стола", "столов", "m"),
    "trash can": ("мусорный бак", "мусорных бака", "мусорных баков", "m"),
    "dog": ("собака", "собаки", "собак", "f"),
    "cat": ("кошка", "кошки", "кошек", "f"),
    "bottle": ("бутылка", "бутылки", "бутылок", "f"),
    "cup": ("чашка", "чашки", "чашек", "f"),
    "backpack": ("рюкзак", "рюкзака", "рюкзаков", "m"),
    "bag": ("сумка", "сумки", "сумок", "f"),
    "laptop": ("ноутбук", "ноутбука", "ноутбуков", "m"),
    "cell phone": ("телефон", "телефона", "телефонов", "m"),
    "keys": ("ключ", "ключа", "ключей", "m"),
    "window": ("окно", "окна", "окон", "n"),
    "tree": ("дерево", "дерева", "деревьев", "n"),
    "elevator": ("лифт", "лифта", "лифтов", "m"),
    "wall": ("стена", "стены", "стен", "f"),
    "book": ("книга", "книги", "книг", "f"),
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
        "still_loading": "Секунду, ещё загружаю модели.",
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
        "forgot_none": "Такого нет в памяти.",
        "remembered_obj": "Запомнила предмет: {name}.",
        "obj_unsure": "Не разглядела предмет, запомнила то, что в центре кадра: {name}.",
        "memory_list": "Люди: {people}. Предметы: {objects}.",
        "memory_empty": "Память пуста.",
        "none_word": "нет",
        "faces_off": "Распознавание лиц недоступно.",
        "paused": "Фоновые подсказки на паузе.",
        "resumed": "Фоновые подсказки включены.",
        "error": "Произошла ошибка.",
        "count": "В кадре {items}.",
        "count_all": "В кадре: {items}.",
        "count_none": "Ничего знакомого не вижу.",
        "left": "слева", "center": "прямо", "right": "справа",
        "very_close": "совсем близко", "close": "близко", "far": "далеко",
        "approx_m": "около {n} {unit}",
    },
    "kk": {
        "loading": "Модельдер жүктелуде, күте тұрыңыз.",
        "still_loading": "Бір сәт, модельдер әлі жүктелуде.",
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
        "forgot_none": "Мұндай жадта жоқ.",
        "remembered_obj": "Зат есте сақталды: {name}.",
        "obj_unsure": "Затты анық көрмедім, кадр ортасындағыны сақтадым: {name}.",
        "memory_list": "Адамдар: {people}. Заттар: {objects}.",
        "memory_empty": "Жад бос.",
        "none_word": "жоқ",
        "faces_off": "Бет тану қолжетімсіз.",
        "paused": "Фондық кеңестер тоқтатылды.",
        "resumed": "Фондық кеңестер қосылды.",
        "error": "Қате орын алды.",
        "count": "Кадрда {items}.",
        "count_all": "Кадрда: {items}.",
        "count_none": "Таныс ештеңе көрінбейді.",
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


def count_phrase(n: int, en: str | None, lang: str, raw: str = "") -> str:
    """«три человека», «одна дверь», «үш адам»; для неизвестного объекта — «зонтики: два»."""
    num = number_words(n, lang)
    if lang == "kk":
        return f"{num} {object_name(en, lang) or raw or en}"
    forms = RU_COUNT_FORMS.get(en)
    if not forms:
        return f"{raw or object_name(en, lang) or en}: {num}"
    one, few, many, gender = forms
    last = n % 10 if not 10 <= n % 100 < 20 else 0
    if gender != "m" and last in (1, 2):
        head, _, tail = num.rpartition(" ")
        tail = {"один": "одна" if gender == "f" else "одно", "два": "две" if gender == "f" else "два"}[tail]
        num = f"{head} {tail}".strip()
    return f"{num} {one if last == 1 else few if last in (2, 3, 4) else many}"


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
