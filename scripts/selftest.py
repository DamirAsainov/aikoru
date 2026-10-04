"""Самопроверка компонентов без микрофона и (опционально) без камеры.

    python scripts/selftest.py            # все проверки
    python scripts/selftest.py tts stt    # выборочно

TTS синтезирует фразы → Vosk распознаёт их обратно (проверка STT и слова
активации), YOLOE и Moondream проверяются на кадре с камеры или на тестовом
изображении ultralytics.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aikoru.__main__ import prepare_env  # noqa: E402
from aikoru.config import load_config  # noqa: E402

log = logging.getLogger("selftest")


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    n = int(len(audio) * sr_to / sr_from)
    return np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)


def get_frame(cfg):
    import cv2

    try:
        from aikoru.camera import Camera

        cam = Camera(cfg["camera"]["index"])
        time.sleep(3.0)  # автоэкспозиция вебкамеры: первые кадры тёмные
        frame = cam.read()
        cam.close()
        log.info("Кадр с камеры: %s, яркость %.0f", frame.shape, frame.mean())
        if frame.mean() < 10:
            raise RuntimeError("кадр слишком тёмный")
        return frame
    except Exception as e:
        log.warning("Камера недоступна (%s) — беру тестовое изображение", e)
        from ultralytics.utils import ASSETS

        return cv2.imread(str(ASSETS / "zidane.jpg"))  # люди с хорошо видимыми лицами


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    prepare_env(cfg["models_dir"], offline=True)
    only = {a for a in sys.argv[1:] if not a.startswith("--")} or {
        "tts", "stt", "commands", "yoloe", "track", "faces", "nllb", "vlm"}
    dev = cfg["device"]
    log.info("Устройство: %s", dev)

    speaker = None
    if "tts" in only or "stt" in only:
        from aikoru.tts import Speaker

        t = cfg["tts"]
        speaker = Speaker(cfg["models_dir"], t["model"], t["voice"], t["sample_rate"], "cpu")
        for lang, text in [("ru", "Осторожно! Машина прямо, около трёх метров."),
                           ("kk", "Абайлаңыз! Көлік алдыңызда, шамамен үш метр.")]:
            t0 = time.time()
            audio = speaker.synth(text, lang)
            log.info("TTS %s: %.1f с аудио за %.2f с", lang, len(audio) / t["sample_rate"], time.time() - t0)
        if "--play" in sys.argv:
            speaker.say("Проверка звука.", "ru")
            speaker.wait_idle()

    if "stt" in only:
        from aikoru.commands import pick_language
        from aikoru.voice import VoiceInput, wake_score

        vc = cfg["voice"]
        vi = VoiceInput(cfg["models_dir"], vc["vosk_models"], vc["wake_words"], vc["wake_threshold"],
                        vc["command_timeout_s"], vc["silence_end_s"], lambda: None, lambda h: None)
        for lang, text in [("ru", "айкору"), ("ru", "что вокруг"), ("ru", "найди дверь"),
                           ("kk", "айналада не бар"), ("kk", "есікті тап")]:
            audio = resample(speaker.synth(text, lang), cfg["tts"]["sample_rate"], 16000)
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
            hyps = vi.recognize(pcm)
            if text == "айкору":
                raw = [vi._new_rec(l) for l in vi.models]
                scores = []
                for rec in raw:
                    rec.AcceptWaveform(pcm)
                    import json
                    heard = json.loads(rec.FinalResult())["text"]
                    scores.append((heard, round(wake_score(heard, vc["wake_words"]), 2)))
                log.info("Слово активации %r → %s", text, scores)
            else:
                cmd = pick_language(hyps)
                log.info("STT %r → %s / %s / target=%s", text, cmd.lang, cmd.intent, cmd.target_en)

    if "commands" in only:
        from aikoru.commands import parse

        for lang, text in [("ru", "что вокруг"), ("ru", "найди скамейку"), ("ru", "где светофор"),
                           ("ru", "что написано на вывеске"), ("ru", "стоп"),
                           ("kk", "айналада не бар"), ("kk", "есікті тап"), ("kk", "бағдаршам қайда"),
                           ("ru", "найди красный зонт")]:
            c = parse(text, lang)
            log.info("parse %-28r → %s target=%s raw=%r", text, c.intent, c.target_en, c.target_raw)

    frame = None
    if only & {"yoloe", "vlm", "faces", "track"}:
        frame = get_frame(cfg)

    if "yoloe" in only:
        from aikoru.detector import Detector

        d = cfg["detector"]
        det = Detector(d["weights"], dev, d["conf"])
        for names in (d["standard_classes"], ["door"], ["person"]):
            t0 = time.time()
            dets = det.detect(frame, names)
            log.info("YOLOE %s… → %s (%.2f с)", names[:3],
                     [(x.name, x.direction, x.distance, x.meters) for x in dets[:5]], time.time() - t0)

    if "track" in only:
        from aikoru.camera import Camera
        from aikoru.detector import Detector
        from aikoru.scene import Scene

        d = cfg["detector"]
        trk = Detector(d["weights"], dev, d["conf"], d["standard_classes"], d["tracker"])
        scene = Scene()
        cam = Camera(cfg["camera"]["index"])
        time.sleep(2)
        dark = cam.read().mean() < 10
        t0, n = time.time(), 0
        while time.time() - t0 < 5:
            f = frame if dark and frame is not None else cam.read()
            dets = trk.track(f)
            scene.update(dets, time.time(), f.shape[1])
            n += 1
        cam.close()
        log.info("Трекинг: %.1f кадров/с, треки: %s", n / 5,
                 [(t.id, t.name, t.hits, t.confirmed) for t in scene.tracks.values()])

    if "faces" in only:
        from aikoru.faces import FaceBook

        fb = FaceBook(cfg["models_dir"], str(Path(cfg["models_dir"]) / "selftest_faces.json"))
        n = fb.remember([frame], "Тест")
        log.info("Лица: сохранено образцов %d; узнаю: %s", n, fb.identify(frame))
        fb.forget(None)

    tr = None
    if "nllb" in only or "vlm" in only:
        from aikoru.translator import Translator

        tr = Translator(cfg["translator"]["path"], dev)
        for text, src, tgt in [("A man is crossing the street near a bus.", "en", "ru"),
                               ("A man is crossing the street near a bus.", "en", "kk"),
                               ("красный зонт", "ru", "en"), ("қызыл қолшатыр", "kk", "en")]:
            t0 = time.time()
            log.info("NLLB %s→%s: %r → %r (%.2f с)", src, tgt, text, tr.translate(text, src, tgt), time.time() - t0)

    if "vlm" in only:
        from aikoru.vlm import SceneDescriber

        v = cfg["vlm"]
        vlm = SceneDescriber(v["path"], dev, v["caption_length"])
        t0 = time.time()
        cap = vlm.describe(frame)
        log.info("Moondream (%.2f с): %s", time.time() - t0, cap)
        log.info("→ ru: %s", tr.translate(cap, "en", "ru"))
        log.info("→ kk: %s", tr.translate(cap, "en", "kk"))

    log.info("Самопроверка завершена")


if __name__ == "__main__":
    main()
