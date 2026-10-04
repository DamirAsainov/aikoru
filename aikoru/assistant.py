"""Оркестратор AIKORU: камера + YOLOE + Moondream + перевод + голос."""
from __future__ import annotations

import logging
import re
import queue
import threading
import time

import cv2

from . import lexicon
from .commands import Command, parse, pick_language
from .tts import ANSWER, DANGER, INFO

log = logging.getLogger(__name__)


class Assistant:
    def __init__(self, cfg: dict, voice_input: bool = True, show: bool = False):
        from .camera import Camera
        from .detector import Detector
        from .translator import Translator
        from .tts import Speaker
        from .vlm import SceneDescriber

        self.cfg = cfg
        self.lang = cfg["default_language"]
        self.show = show
        dev = cfg["device"]
        t = cfg["tts"]

        log.info("Загрузка TTS (Silero)…")
        self.tts = Speaker(cfg["models_dir"], t["model"], t["voice"], t["sample_rate"], "cpu")
        self.tts.say(self.p("loading"), self.lang)

        d = cfg["detector"]
        log.info("Загрузка YOLOE…")
        # Два экземпляра: трекер с постоянным набором классов (смена классов
        # сбросила бы треки) и детектор для запросов «найди …»
        self.tracker = Detector(d["weights"], dev, d["conf"], d["standard_classes"], d["tracker"])
        self.detector = Detector(d["weights"], dev, d["conf"])
        self.faces = None
        f = cfg.get("faces", {})
        if f.get("enabled"):
            from .faces import FaceBook

            try:
                self.faces = FaceBook(cfg["models_dir"], f["db"], f["threshold"])
            except FileNotFoundError as e:
                log.warning("Распознавание лиц отключено: %s", e)
        log.info("Загрузка переводчика (NLLB)…")
        self.translator = Translator(cfg["translator"]["path"], dev)
        log.info("Загрузка VLM (Moondream)…")
        v = cfg["vlm"]
        self.vlm = SceneDescriber(v["path"], dev, v["caption_length"])
        log.info("Прогрев моделей…")  # первый вызов Moondream/YOLOE в разы медленнее
        import numpy as np

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.vlm.describe(blank)
        self.detector.detect(blank, ["door"])
        self.tracker.track(blank)
        log.info("Открываю камеру…")
        c = cfg["camera"]
        self.camera = Camera(c["index"], c["width"], c["height"])

        self.voice = None
        if voice_input:
            from .voice import VoiceInput

            vc = cfg["voice"]
            self.voice = VoiceInput(
                cfg["models_dir"], vc["vosk_models"], vc["wake_words"], vc["wake_threshold"],
                vc["command_timeout_s"], vc["silence_end_s"],
                on_wake=self._on_wake, on_command=self._on_voice_command,
            )

        from .scene import Scene

        self.paused = False
        self.scene = Scene()
        self._commands: queue.Queue[Command | None] = queue.Queue()
        self._busy = threading.Event()   # идёт обработка команды
        self._running = False
        self._last_dets = []

    def p(self, key: str, lang: str | None = None) -> str:
        return lexicon.PHRASES[lang or self.lang][key]

    # --- запуск -----------------------------------------------------------
    def run(self) -> None:
        self._running = True
        threading.Thread(target=self._scan_loop, name="scan", daemon=True).start()
        threading.Thread(target=self._command_loop, name="commands", daemon=True).start()
        if self.voice:
            self.voice.start()
        self.tts.say(self.p("ready"), self.lang)
        try:
            if self.voice:
                self._idle_loop()
            else:
                self._console_loop()
        finally:
            self._running = False
            if self.voice:
                self.voice.stop()
            self.camera.close()
            cv2.destroyAllWindows()

    def _idle_loop(self) -> None:
        while self._running:
            if self.show:
                self._show_frame()
            else:
                time.sleep(0.1)

    def _console_loop(self) -> None:
        """Режим отладки без микрофона: команды вводятся текстом. Префикс kk: — казахский."""
        print("Вводите команды (например: «что вокруг», «найди дверь», «kk: есікті тап»). Ctrl+C — выход.")
        while self._running:
            try:
                line = input("> ").strip()
            except EOFError:  # ввод из файла/пайпа закончился — дождаться ответов и выйти
                while not self._commands.empty() or self._busy.is_set():
                    time.sleep(0.2)
                self.tts.wait_idle()
                return
            if not line:
                continue
            lang = self.lang
            if line[:3] in ("ru:", "kk:"):
                lang, line = line[:2], line[3:].strip()
            self._commands.put(parse(line, lang))

    def _show_frame(self) -> None:
        frame = self.camera.read()
        idents = {t.id: t.identity for t in list(self.scene.tracks.values())}
        for det in self._last_dets:
            x1, y1, x2, y2 = map(int, det.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            who = idents.get(det.track_id) or det.name
            label = f"#{det.track_id} {who} {det.meters or ''}m {det.direction}"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.imshow("AIKORU", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            self._running = False

    # --- голос ------------------------------------------------------------
    def _on_wake(self) -> None:
        self._busy.set()
        self.tts.beep()

    def _on_voice_command(self, hyps) -> None:
        if not hyps:
            self.tts.say(self.p("not_heard"), self.lang, ANSWER)
            self._busy.clear()
            return
        self._commands.put(pick_language(hyps))

    # --- обработка команд -------------------------------------------------
    def _command_loop(self) -> None:
        while self._running:
            try:
                cmd = self._commands.get(timeout=0.5)
            except queue.Empty:
                continue
            self._busy.set()
            try:
                self.handle(cmd)
            except Exception:
                log.exception("Ошибка обработки команды %s", cmd)
                self.tts.say(self.p("error", cmd.lang), cmd.lang, ANSWER)
            finally:
                self._busy.clear()

    def handle(self, cmd: Command) -> None:
        log.info("Команда: %s", cmd)
        lang = self.lang = cmd.lang
        if cmd.intent == "pause":
            self.paused = True
            self.tts.say(self.p("paused"), lang, ANSWER)
        elif cmd.intent == "resume":
            self.paused = False
            self.tts.say(self.p("resumed"), lang, ANSWER)
        elif cmd.intent == "describe":
            self.tts.say(self.p("looking"), lang, ANSWER)
            caption = self.vlm.describe(self.camera.read())
            log.info("Moondream: %s", caption)
            self.tts.say(self.translator.translate(caption, "en", lang), lang, ANSWER)
        elif cmd.intent == "find":
            self._find(cmd)
        elif cmd.intent in ("who", "remember", "forget"):
            if self.faces is None:
                self.tts.say(self.p("faces_off", lang), lang, ANSWER)
            else:
                getattr(self, f"_{cmd.intent}")(cmd)
        elif cmd.intent == "ask":
            self.tts.say(self.p("looking"), lang, ANSWER)
            question = self.translator.translate(cmd.text, lang, "en")
            answer = self.vlm.ask(self.camera.read(), question)
            log.info("Вопрос: %s → %s", question, answer)
            self.tts.say(self.translator.translate(answer, "en", lang), lang, ANSWER)
        else:
            self.tts.say(self.p("not_heard"), lang, ANSWER)

    def _find(self, cmd: Command) -> None:
        lang = cmd.lang
        if cmd.target_en:
            target_en = cmd.target_en
            spoken = lexicon.object_name(target_en, lang)
        else:
            target_en = self.translator.translate(cmd.target_raw, lang, "en").lower().strip(" .!?")
            target_en = re.sub(r"^(the|a|an)\s+", "", target_en)
            spoken = cmd.target_raw
        log.info("Поиск: %r → %r", cmd.target_raw, target_en)
        dets = self.detector.detect(self.camera.read(), [target_en], conf=0.2)
        self._last_dets = dets
        if not dets:
            obj = spoken if lang == "ru" else spoken.capitalize()
            self.tts.say(self.p("not_found", lang).format(obj=obj), lang, ANSWER)
            return
        d = dets[0]
        self.tts.say(self.p("found", lang).format(
            obj=spoken.capitalize(), dir=self.p(d.direction, lang),
            dist=lexicon.distance_phrase(d.distance, d.meters, lang),
        ), lang, ANSWER)

    # --- лица -----------------------------------------------------------
    def _who(self, cmd: Command) -> None:
        lang = cmd.lang
        found = self.faces.identify_all(self.camera.read())
        if not found:
            # лиц не видно — хотя бы скажем, сколько людей трекер видит рядом
            people = [t for t in self.scene.tracks.values() if t.name == "person" and t.confirmed]
            if not people:
                self.tts.say(self.p("who_none", lang), lang, ANSWER)
                return
            found = [(t.identity, t.cx) for t in people]
        items = []
        for name, cx in sorted(found, key=lambda x: x[1]):
            direction = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"
            items.append(f"{name or self.p('unknown_person', lang)} {self.p(direction, lang)}")
        self.tts.say(lexicon.sentence("; ".join(items)) + ".", lang, ANSWER)

    def _remember(self, cmd: Command) -> None:
        lang = cmd.lang
        if not cmd.name:
            self.tts.say(self.p("need_name", lang), lang, ANSWER)
            return
        frames = []
        for _ in range(self.cfg["faces"]["remember_samples"]):
            frames.append(self.camera.read())
            time.sleep(0.3)
        if self.faces.remember(frames, cmd.name) == 0:
            self.tts.say(self.p("no_face", lang), lang, ANSWER)
            return
        # человек, который сейчас ближе всех, теперь знакомый — не объявлять его заново
        people = [t for t in self.scene.tracks.values() if t.name == "person"]
        if people:
            nearest = min(people, key=lambda t: t.det.meters or 99)
            nearest.identity, nearest.announced = cmd.name, True
        self.tts.say(self.p("remembered", lang).format(name=cmd.name), lang, ANSWER)

    def _forget(self, cmd: Command) -> None:
        lang = cmd.lang
        removed = self.faces.forget(cmd.name or None)
        if not removed:
            self.tts.say(self.p("forgot_none", lang), lang, ANSWER)
            return
        for t in self.scene.tracks.values():
            if t.identity in removed:
                t.identity = None
        self.tts.say(self.p("forgot", lang).format(names=", ".join(removed)), lang, ANSWER)

    # --- фоновый трекинг -------------------------------------------------
    def _scan_loop(self) -> None:
        interval = self.cfg["detector"]["scan_interval_s"]
        while self._running:
            started = time.time()
            try:
                self._scan_once()
            except Exception:
                log.exception("Ошибка фонового трекинга")
            time.sleep(max(0.0, interval - (time.time() - started)))

    def _scan_once(self) -> None:
        """Трекинг идёт всегда (чтобы не терять треки), а говорим только об изменениях."""
        frame = self.camera.read()
        dets = self.tracker.track(frame)
        self._last_dets = dets
        now = time.time()
        gone = self.scene.update(dets, now, frame.shape[1])
        if self.paused:
            return
        lang = self.lang
        d = self.cfg["detector"]

        new_strangers = []
        for t in self.scene.visible(now):
            if not t.confirmed:
                continue
            det = t.det
            where = {"dir": self.p(det.direction, lang),
                     "dist": lexicon.distance_phrase(det.distance, det.meters, lang)}

            # 1) Опасность прямо по курсу: один раз на трек и ещё раз, если подошла вплотную
            if det.name in d["danger_classes"] and det.direction == "center"                     and det.distance in ("very_close", "close")                     and t.danger_level != det.distance and t.danger_level != "very_close":
                t.danger_level = det.distance
                t.announced = True
                self.tts.say(self.p("danger", lang).format(
                    obj=lexicon.sentence(lexicon.object_name(det.name, lang)), **where), lang, DANGER)
                continue

            # 2) Люди: сначала пытаемся узнать лицо
            if det.name == "person":
                if self._person_event(t, frame, now, where, lang):
                    new_strangers.append(t)
                continue

            # 3) Новый объект рядом (дальние объявим, когда подойдут ближе)
            if not t.announced and det.distance != "far":
                t.announced = True
                self.tts.say(lexicon.sentence(self.p("appeared", lang).format(
                    obj=lexicon.object_name(det.name, lang), **where)), lang, INFO)

        # Незнакомцев, появившихся одновременно, объявляем одной фразой
        if len(new_strangers) == 1:
            det = new_strangers[0].det
            self.tts.say(lexicon.sentence(self.p("new_person", lang).format(
                dir=self.p(det.direction, lang),
                dist=lexicon.distance_phrase(det.distance, det.meters, lang))), lang, INFO)
        elif new_strangers:
            det = min((t.det for t in new_strangers), key=lambda x: x.meters or 99)
            self.tts.say(self.p("new_people", lang).format(
                n=lexicon.number_words(len(new_strangers), lang), dir=self.p(det.direction, lang),
                dist=lexicon.distance_phrase(det.distance, det.meters, lang)), lang, INFO)

        # 4) Знакомый человек ушёл
        for t in gone:
            if t.identity and t.announced:
                self.tts.say(self.p("person_left", lang).format(name=t.identity), lang, INFO)

    def _person_event(self, t, frame, now: float, where: dict, lang: str) -> bool:
        """Узнаёт лицо; True — это новый незнакомец, о котором пора сообщить."""
        f = self.cfg["faces"]
        if self.faces and t.identity is None and t.face_tries < f["max_tries"]                 and now - t.last_face_try >= f["retry_interval_s"]:
            t.face_tries += 1
            t.last_face_try = now
            x1, y1, x2, y2 = map(int, t.det.box)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                name, _ = self.faces.identify(crop)
                if name:
                    t.identity = name
                    t.announced = True
                    self.tts.say(lexicon.sentence(self.p("known_person", lang).format(
                        name=name, **where)), lang, ANSWER)
                    return False
        if t.announced or t.identity:
            return False
        # незнакомец: объявляем, когда он рядом и лицо за отведённое время не узнали
        waited = now - t.first_seen >= f["unknown_after_s"] or not self.faces
        if t.det.distance != "far" and waited:
            t.announced = True
            return True
        return False
