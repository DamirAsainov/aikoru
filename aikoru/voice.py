"""Голосовой ввод: слово активации (Vosk ru+kk) → запись команды (Silero VAD) → распознавание.

Оба распознавателя Vosk (ru и kk) слушают параллельно; язык команды выбирается
по тому, чей результат лучше (см. commands.pick_language).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SR = 16000
CHUNK = 512  # Silero VAD требует 512 сэмплов при 16 кГц


@dataclass
class Hypothesis:
    lang: str
    text: str
    conf: float


def _normalize(s: str) -> str:
    return s.lower().replace("ё", "е").replace(" ", "")


def wake_score(text: str, wake_words: list[str]) -> float:
    """Лучшее нечёткое совпадение окон из 1–3 слов с одним из вариантов слова активации."""
    words = text.lower().split()
    targets = [_normalize(w) for w in wake_words]
    best = 0.0
    for i in range(len(words)):
        for n in (1, 2, 3):
            if i + n > len(words):
                break
            cand = _normalize("".join(words[i:i + n]))
            for t in targets:
                best = max(best, SequenceMatcher(None, cand, t).ratio())
    return best


def strip_wake(text: str, wake_words: list[str], threshold: float) -> str:
    """Убирает из начала команды хвост слова активации."""
    words = text.split()
    while words and wake_score(words[0], wake_words) >= threshold - 0.1:
        words.pop(0)
    return " ".join(words)


class VoiceInput:
    def __init__(
        self,
        models_dir: str,
        vosk_models: dict[str, str],
        wake_words: list[str],
        wake_threshold: float,
        command_timeout_s: float,
        silence_end_s: float,
        on_wake: Callable[[], None],
        on_command: Callable[[list[Hypothesis]], None],
    ):
        import vosk
        from silero_vad import VADIterator, load_silero_vad

        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self.models = {}
        for lang, name in vosk_models.items():
            path = Path(models_dir) / "vosk" / name
            if not path.exists():
                raise FileNotFoundError(f"{path} не найден — запустите scripts/download_models.py")
            self.models[lang] = vosk.Model(str(path))
        self.wake_words = wake_words
        self.wake_threshold = wake_threshold
        self.command_timeout_s = command_timeout_s
        self.vad = VADIterator(
            load_silero_vad(), sampling_rate=SR, min_silence_duration_ms=int(silence_end_s * 1000)
        )
        self.on_wake = on_wake
        self.on_command = on_command
        self._audio: queue.Queue[bytes] = queue.Queue()
        self._running = False

    # --- распознавание ----------------------------------------------------
    def _new_rec(self, lang: str, words: bool = False):
        rec = self._vosk.KaldiRecognizer(self.models[lang], SR)
        rec.SetWords(words)
        return rec

    def recognize(self, pcm: bytes) -> list[Hypothesis]:
        """Распознаёт фрагмент всеми моделями. pcm — int16 mono 16 кГц."""
        hyps = []
        for lang in self.models:
            rec = self._new_rec(lang, words=True)
            rec.AcceptWaveform(pcm)
            res = json.loads(rec.FinalResult())
            words = res.get("result", [])
            conf = float(np.mean([w["conf"] for w in words])) if words else 0.0
            text = strip_wake(res.get("text", ""), self.wake_words, self.wake_threshold)
            hyps.append(Hypothesis(lang, text, conf))
        log.info("Распознано: %s", [(h.lang, h.text, round(h.conf, 2)) for h in hyps])
        return hyps

    # --- главный цикл -----------------------------------------------------
    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, name="voice", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def _callback(self, indata, frames, t, status) -> None:
        self._audio.put(bytes(indata))

    def _loop(self) -> None:
        import torch

        wake_recs = {lang: self._new_rec(lang) for lang in self.models}
        state = "wake"
        buf: list[bytes] = []
        started = False
        t0 = 0.0

        with sd.RawInputStream(samplerate=SR, blocksize=CHUNK, channels=1, dtype="int16",
                               callback=self._callback):
            log.info("Микрофон включён, жду слово активации")
            while self._running:
                try:
                    chunk = self._audio.get(timeout=0.5)
                except queue.Empty:
                    continue

                if state == "wake":
                    for lang, rec in wake_recs.items():
                        if rec.AcceptWaveform(chunk):
                            text = json.loads(rec.Result()).get("text", "")
                        else:
                            text = json.loads(rec.PartialResult()).get("partial", "")
                        if text and wake_score(text, self.wake_words) >= self.wake_threshold:
                            log.info("Слово активации (%s): %r", lang, text)
                            for r in wake_recs.values():
                                r.Reset()
                            self.vad.reset_states()
                            buf, started, t0 = [], False, time.time()
                            state = "command"
                            self.on_wake()
                            break
                    continue

                # state == "command"
                buf.append(chunk)
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                event = self.vad(torch.from_numpy(samples))
                if event and "start" in event:
                    started = True
                elapsed = time.time() - t0
                finished = started and event is not None and "end" in event
                timed_out = elapsed > self.command_timeout_s + (8 if started else 0)
                if finished or timed_out:
                    hyps = self.recognize(b"".join(buf)) if started else []
                    state = "wake"
                    self.vad.reset_states()
                    self.on_command(hyps)
