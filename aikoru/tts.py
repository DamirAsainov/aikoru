"""Синтез речи Silero v5 CIS (один голос для ru и kk) с очередью приоритетов.

Для русского текста ударения расставляет silero-stress (модель nostress
требует их только для славянских языков).

Приоритеты: 0 — опасность (прерывает фоновые подсказки), 1 — ответ на команду,
2 — фоновая подсказка (устаревшие фоновые сообщения выбрасываются).
"""
from __future__ import annotations

import itertools
import logging
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from .lexicon import spell_digits

log = logging.getLogger(__name__)

DANGER, ANSWER, INFO = 0, 1, 2
INFO_MAX_AGE_S = 4.0
SILERO_URL = "https://models.silero.ai/models/tts/ru/{model}.pt"
SPEAKER_PREFIX = {"ru": "ru", "kk": "kaz"}


def tts_model_path(models_dir: str, model: str) -> Path:
    return Path(models_dir) / "silero" / f"{model}.pt"


class Speaker:
    def __init__(self, models_dir: str, model: str, voice: str, sample_rate: int, device: str):
        import torch

        path = tts_model_path(models_dir, model)
        if not path.exists():
            raise FileNotFoundError(f"{path} не найден — запустите scripts/download_models.py")
        self.model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        self.model.to(torch.device(device))
        self.speakers = {lang: f"{p}_{voice}" for lang, p in SPEAKER_PREFIX.items()}
        missing = [s for s in self.speakers.values() if s not in self.model.speakers]
        if missing:
            raise ValueError(f"Голоса {missing} не найдены. Доступные: {sorted(self.model.speakers)}")
        from silero_stress import load_accentor

        self.ru_accentor = load_accentor(lang="ru")
        self.sample_rate = sample_rate
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._playing_priority: int | None = None
        self._interrupt = threading.Event()
        self.listeners: list = []  # callback(text, lang, priority) — например, веб-интерфейс
        self._thread = threading.Thread(target=self._loop, name="tts", daemon=True)
        self._thread.start()

    # --- публичный API ---------------------------------------------------
    def say(self, text: str, lang: str, priority: int = ANSWER) -> None:
        log.info("[%s/p%d] %s", lang, priority, text)
        for cb in self.listeners:
            try:
                cb(text, lang, priority)
            except Exception:
                log.exception("Ошибка подписчика TTS")
        self._q.put((priority, next(self._seq), time.time(), "text", (text, lang)))
        if self._playing_priority is not None and priority < self._playing_priority:
            self._interrupt.set()

    def beep(self, freq: float = 880.0, dur: float = 0.12) -> None:
        self._q.put((DANGER, next(self._seq), time.time(), "beep", (freq, dur)))

    def synth(self, text: str, lang: str) -> np.ndarray:
        text = self._clean(spell_digits(text, lang))
        if not text:
            return np.zeros(0, dtype=np.float32)
        if lang == "ru":
            text = self.ru_accentor(text)
        audio = self.model.apply_tts(text=text, speaker=self.speakers[lang], sample_rate=self.sample_rate)
        return audio.cpu().numpy().astype(np.float32)

    def wait_idle(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while (not self._q.empty() or self._playing_priority is not None) and time.time() < deadline:
            time.sleep(0.05)

    # --- внутреннее ------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        # Silero cyrillic не знает латиницу и редкие символы
        text = re.sub(r"[A-Za-z]+", " ", text)
        text = re.sub(r"[^\w\s.,!?:;\-—()«»]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _loop(self) -> None:
        while True:
            priority, _, created, kind, payload = self._q.get()
            if priority == INFO and time.time() - created > INFO_MAX_AGE_S:
                continue
            try:
                if kind == "beep":
                    freq, dur = payload
                    t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
                    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                else:
                    audio = self.synth(*payload)
                self._play(audio, priority)
            except Exception:
                log.exception("Ошибка синтеза/воспроизведения")

    def _play(self, audio: np.ndarray, priority: int) -> None:
        if audio.size == 0:
            return
        self._interrupt.clear()
        self._playing_priority = priority
        try:
            sd.play(audio, self.sample_rate)
            end = time.time() + len(audio) / self.sample_rate + 0.2
            while time.time() < end:
                if self._interrupt.is_set():
                    sd.stop()
                    break
                time.sleep(0.03)
            else:
                sd.wait()
        finally:
            self._playing_priority = None
