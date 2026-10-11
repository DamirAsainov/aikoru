"""Память личных предметов: визуальные эмбеддинги YOLOE (поиск по фото-образцу).

«Запомни мои ключи» → рамка предмета на нескольких кадрах → один 512-d эмбеддинг
визуального промпта (SAVPE). «Найди мои ключи» → YOLOE ищет по этому эмбеддингу
вместо текста: один обычный проход детектора, без CLIP и без повторного кодирования
образца. База хранится локально в data/objects.json: {имя: {cls, emb, n}}.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Притяжательные и служебные слова не участвуют в сравнении имён
SKIP_WORDS = {"мой", "моя", "мою", "мои", "моё", "мое", "моего", "моей", "моих", "свой", "свою", "свои",
              "менің", "өзімнің", "the", "my"}


def _stems(text: str) -> list[str]:
    """Грубый стемминг для ru/kk: первые 4 буквы слова («ключи», «ключ» → «ключ»)."""
    words = re.findall(r"[\w-]+", text.lower().replace("ё", "е"))
    return [w[:4] for w in words if w not in SKIP_WORDS]


class ObjectBook:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.db: dict[str, dict] = {}
        self._pe: dict[str, object] = {}  # готовые тензоры (1, 1, D) для детектора
        if self.db_path.exists():
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
            self.db = {n: {**v, "emb": np.array(v["emb"], dtype=np.float32)} for n, v in raw.items()}
        log.info("Предметов в памяти: %d (%s)", len(self.db), ", ".join(self.db) or "—")

    def remember(self, name: str, cls: str, emb: np.ndarray, samples: int) -> None:
        """Повторное «запомни» уточняет образец: взвешенное среднее с прежними кадрами."""
        with self._lock:
            old = self.db.get(name)
            if old is not None:
                emb = old["emb"] * old["n"] + emb * samples
                emb /= np.linalg.norm(emb) + 1e-9
                samples += old["n"]
            self.db[name] = {"cls": cls, "emb": emb.astype(np.float32), "n": samples}
            self._pe.pop(name, None)
            self._save()

    def match(self, text: str) -> str | None:
        """Имя предмета, упомянутого в фразе («где мои ключи» → «ключи»), или None."""
        query = set(_stems(text))
        best, best_score = None, 0.0
        for name in list(self.db):  # db может меняться из веб-интерфейса
            stems = _stems(name)
            if not stems:
                continue
            score = sum(s in query for s in stems) / len(stems)
            if score > best_score:
                best, best_score = name, score
        return best if best_score >= 0.5 else None

    def pe(self, name: str):
        """Эмбеддинг для Detector.detect; один и тот же объект, пока образец не изменился."""
        if name not in self._pe:
            import torch

            self._pe[name] = torch.from_numpy(self.db[name]["emb"])[None, None]
        return self._pe[name]

    def forget(self, name: str | None) -> list[str]:
        """Удаляет предмет по имени (нечётко) или все, если name is None."""
        with self._lock:
            if name is None:
                removed = list(self.db)
            else:
                hit = self.match(name)
                removed = [hit] if hit else []
            for n in removed:
                del self.db[n]
                self._pe.pop(n, None)
            self._save()
        return removed

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {n: {"cls": v["cls"], "n": v["n"], "emb": v["emb"].round(5).tolist()} for n, v in self.db.items()}
        self.db_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
