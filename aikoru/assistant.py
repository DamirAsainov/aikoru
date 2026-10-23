"""Оркестратор AIKORU: камера + YOLOE + VLM + перевод + голос."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import Counter
from pathlib import Path

import cv2

from . import lexicon
from .commands import Command, parse, pick_language
from .faces import is_sharp
from .tts import ANSWER, DANGER, INFO

log = logging.getLogger(__name__)


class Assistant:
    def __init__(self, cfg: dict, voice_input: bool = True, show: bool = False, web: bool = True):
        from .camera import Camera
        from .detector import Detector
        from .tts import Speaker

        self.cfg = cfg
        self.lang = cfg["default_language"]
        self.show = show
        dev = cfg["device"]
        t = cfg["tts"]

        log.info("Загрузка TTS (Silero)…")
        self.tts = Speaker(cfg["models_dir"], t["model"], t["voice"], t["sample_rate"], "cpu")
        self.tts.say(self.p("loading"), self.lang)

        # камера открывается 2–4 с (DirectShow): параллельно с загрузкой YOLOE
        c = cfg["camera"]
        opened: dict = {}

        def open_camera() -> None:
            try:
                opened["camera"] = Camera(c["index"], c["width"], c["height"], c.get("fps", 15))
            except (RuntimeError, cv2.error) as e:  # ошибка всплывёт в основном потоке
                opened["error"] = e

        camera_thread = threading.Thread(target=open_camera, name="camera-open", daemon=True)
        camera_thread.start()

        d = cfg["detector"]
        log.info("Загрузка YOLOE…")
        # Два экземпляра: трекер с постоянным набором классов (смена классов
        # сбросила бы треки) и детектор для запросов «найди …»
        imgsz = d.get("imgsz", 640)
        self.tracker = self._make_tracker(d, dev, imgsz, cfg["camera"])
        self.detector = Detector(d["weights"], dev, d["conf"], imgsz=imgsz)
        self.faces = None
        f = cfg.get("faces", {})
        if f.get("enabled"):
            from .faces import FaceBook

            try:
                self.faces = FaceBook(cfg["models_dir"], f["db"], f["threshold"])
            except FileNotFoundError as e:
                log.warning("Распознавание лиц отключено: %s", e)
        self.objects = None
        if cfg.get("objects", {}).get("enabled"):
            from .objects import ObjectBook

            self.objects = ObjectBook(cfg["objects"]["db"])
        # NLLB и VLM нужны только командам: грузятся в фоне, а «Система готова»
        # звучит, как только работают трекинг, камера и голос. Поток стартует
        # после YOLOE, чтобы импорты ultralytics и transformers не шли одновременно.
        self._translator = None
        self._vlm = None
        self._translator_ready = threading.Event()
        self._vlm_ready = threading.Event()
        self._said_loading = 0.0
        threading.Thread(target=self._load_heavy, name="load-heavy", daemon=True).start()

        log.info("Прогрев YOLOE…")  # первый вызов в разы медленнее
        import numpy as np

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.detect(blank, ["door"])
        self.tracker.track(blank)
        if self.objects is not None:  # первый визуальный промпт строит predictor (~1 с)
            self.detector.visual_embedding([blank], [(200, 150, 440, 330)])
        camera_thread.join()
        if "error" in opened:
            raise opened["error"]
        self.camera = opened["camera"]
        self.motion = None
        # ответы VLM по вопросу ("" — описание сцены): (время кадра, текст на английском)
        self._answers: dict[str, tuple[float, str]] = {}
        m = cfg.get("motion", {})
        if m.get("enabled", True):
            from .motion import MotionGate

            self.motion = MotionGate(m.get("width", 160), m.get("local", 0.003),
                                     m.get("global", 0.15), m.get("history", 300),
                                     m.get("var_threshold", 25))
            self.camera.motion = self.motion

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
        d = cfg["detector"]
        self.scene = Scene(close_is_change=self.motion is None)
        self.scan_mode = "active"
        self._commands: queue.Queue[Command | None] = queue.Queue()
        self._busy = threading.Event()   # идёт обработка команды
        self._running = False
        self._last_dets = []
        self.web = None
        w = cfg.get("web", {})
        if web and w.get("enabled", True):
            from .web import WebUI

            self.web = WebUI(self, w.get("host", "127.0.0.1"), w.get("port", 8765))

    def _load_heavy(self) -> None:
        """Фоновая загрузка: сначала NLLB (нужен и для «найди …»), потом VLM с прогревом."""
        import numpy as np

        from .translator import Translator
        from .vlm import LlamaCppDescriber, SceneDescriber

        cfg = self.cfg
        try:
            log.info("Загрузка переводчика (NLLB) в фоне…")
            self._translator = Translator(cfg["translator"]["path"], cfg["translator"]["device"],
                                         cfg["translator"].get("threads", 0))
        except Exception:
            log.exception("Переводчик не загрузился")
        finally:
            self._translator_ready.set()
        try:
            v = cfg["vlm"]
            if v.get("backend") == "llamacpp":
                log.info("Загрузка VLM (%s, llama.cpp) в фоне…", Path(v["gguf"]).name)
                vlm = LlamaCppDescriber(v["server"], v["gguf"], v["mmproj"], v["image_tokens"],
                                        v["max_new_tokens"], v.get("threads", 0), v.get("port", 8766))
            else:
                log.info("Загрузка VLM (%s, %s) в фоне…", Path(v["path"]).name, v["device"])
                vlm = SceneDescriber(v["path"], v["device"], v["image_tokens"], v["max_new_tokens"],
                                     v.get("threads", 0))
            vlm.describe(np.zeros((480, 640, 3), dtype=np.uint8))  # первый вызов в разы медленнее
            self._vlm = vlm
            log.info("VLM готова")
        except Exception:
            log.exception("VLM не загрузилась")
        finally:
            self._vlm_ready.set()

    @property
    def translator(self):
        return self._wait_model(self._translator_ready, "_translator")

    @property
    def vlm(self):
        return self._wait_model(self._vlm_ready, "_vlm")

    def _wait_model(self, ready: threading.Event, attr: str):
        """Команда пришла раньше, чем модель догрузилась: предупредить и дождаться."""
        if not ready.is_set():
            if time.time() - self._said_loading > 10:  # одна фраза на команду, а не на модель
                self._said_loading = time.time()
                self.tts.say(self.p("still_loading"), self.lang, ANSWER)
            ready.wait()
        model = getattr(self, attr)
        if model is None:
            raise RuntimeError(f"Модель {attr.lstrip('_')} не загрузилась, см. лог выше")
        return model

    @staticmethod
    def _make_tracker(d: dict, dev: str, imgsz: int, cam: dict):
        """Трекер: на CPU — OpenVINO (i5-10400: ~50 мс вместо ~86 мс у PyTorch), иначе PyTorch.

        Вход OpenVINO статичный, поэтому берётся прямоугольник под кадр камеры
        (640×480 → 512×384), а не квадрат: на четверть меньше работы.
        Без OMP_WAIT_POLICY=PASSIVE (config.prepare_openmp) OpenVINO медленнее PyTorch.
        """
        from .config import openmp_passive
        from .detector import MIN_CALIB_FRAMES, Detector, calib_frames, openvino_available

        backend = d.get("backend", "auto")
        if backend == "auto":
            backend = "openvino" if dev == "cpu" and openvino_available() and openmp_passive() else "torch"
        elif backend == "openvino" and not openmp_passive():
            log.warning("OMP_WAIT_POLICY не PASSIVE: OpenVINO будет медленнее PyTorch")
        if backend == "openvino":
            h = round(imgsz * cam["height"] / cam["width"] / 32) * 32
            calib = None
            if d.get("int8"):
                calib = d.get("calib_dir")
                n = len(calib_frames(calib))
                if n < MIN_CALIB_FRAMES:
                    log.warning("INT8: в %s %d кадров из %d нужных — трекер в FP32 "
                                "(сначала scripts/capture_calib.py)", calib, n, MIN_CALIB_FRAMES)
                    calib = None
            try:
                return Detector(d["weights"], dev, d["conf"], d["standard_classes"], d["tracker"],
                                (h, imgsz), backend="openvino", calib_dir=calib)
            except Exception:
                log.exception("OpenVINO недоступен, трекер на PyTorch")
        return Detector(d["weights"], dev, d["conf"], d["standard_classes"], d["tracker"], imgsz)

    def p(self, key: str, lang: str | None = None) -> str:
        return lexicon.PHRASES[lang or self.lang][key]

    # --- запуск -----------------------------------------------------------
    def run(self) -> None:
        self._running = True
        threading.Thread(target=self._scan_loop, name="scan", daemon=True).start()
        threading.Thread(target=self._command_loop, name="commands", daemon=True).start()
        if self.voice:
            self.voice.start()
        if self.web:
            self.web.start()
            print(f"Веб-интерфейс: {self.web.url}")
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
            if self.web:
                self.web.stop()
            self.camera.close()
            if hasattr(self._vlm, "close"):
                self._vlm.close()  # llama-server — отдельный процесс
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
        if self.web:
            self.web.bus.publish("command", text=cmd.text, lang=cmd.lang, intent=cmd.intent)
        lang = self.lang = cmd.lang
        if cmd.intent == "pause":
            self.paused = True
            self.tts.say(self.p("paused"), lang, ANSWER)
        elif cmd.intent == "resume":
            self.paused = False
            self.tts.say(self.p("resumed"), lang, ANSWER)
        elif cmd.intent == "describe":
            last = self._answers.get("")
            if last and self._scene_static_since(last[0]):
                log.info("Сцена не менялась с прошлого описания — повторяю ответ")
                self.tts.say(self.translator.translate(last[1], "en", lang), lang, ANSWER)
                return
            self.tts.say(self.p("looking"), lang, ANSWER)
            started = time.time()
            caption = self.vlm.describe(self.camera.read())
            self._answers[""] = (started, caption)
            log.info("VLM: %s", caption)
            self.tts.say(self.translator.translate(caption, "en", lang), lang, ANSWER)
        elif cmd.intent == "find":
            self._find(cmd)
        elif cmd.intent == "count":
            self._count(cmd)
        elif cmd.intent == "who" and self.faces is None                 or cmd.intent in ("remember", "forget", "memory") and self.faces is None and self.objects is None:
            self.tts.say(self.p("faces_off", lang), lang, ANSWER)
        elif cmd.intent in ("who", "remember", "forget", "memory"):
            getattr(self, f"_{cmd.intent}")(cmd)
        elif cmd.intent == "ask":
            question = self.translator.translate(cmd.text, lang, "en")
            last = self._answers.get(question)
            if last and self._scene_static_since(last[0]):
                log.info("Сцена не менялась — повторяю ответ на «%s»", question)
                self.tts.say(self.translator.translate(last[1], "en", lang), lang, ANSWER)
                return
            self.tts.say(self.p("looking"), lang, ANSWER)
            started = time.time()
            answer = self.vlm.ask(self.camera.read(), question)
            if len(self._answers) >= 16:
                self._answers.pop(next(iter(self._answers)))
            self._answers[question] = (started, answer)
            log.info("Вопрос: %s → %s", question, answer)
            self.tts.say(self.translator.translate(answer, "en", lang), lang, ANSWER)
        else:
            self.tts.say(self.p("not_heard"), lang, ANSWER)

    def _scene_static_since(self, ts: float, max_age_s: float = 120.0) -> bool:
        """С кадра ts в сцене не было движения и изменений — прошлый ответ VLM ещё верен."""
        m = self.motion
        return (m is not None and m.last_local < ts and time.time() - ts < max_age_s
                and not self.scene.changed)

    def _to_english(self, text: str, lang: str) -> str:
        en = self.translator.translate(text, lang, "en").lower().strip(" .!?")
        return re.sub(r"^(the|a|an|my)\s+", "", en)

    def _find(self, cmd: Command) -> None:
        lang = cmd.lang
        mine = self.objects.match(cmd.target_raw or cmd.text) if self.objects else None
        if mine:
            # свой предмет: ищем по запомненному визуальному образцу, а не по тексту
            target_en, spoken, pe = self.objects.db[mine]["cls"], mine, self.objects.pe(mine)
            conf = self.cfg["objects"]["conf"]
        elif cmd.target_en:
            target_en, pe, conf = cmd.target_en, None, 0.2
            spoken = lexicon.object_name(target_en, lang)
        else:
            target_en, pe, conf = self._to_english(cmd.target_raw, lang), None, 0.2
            spoken = cmd.target_raw
        log.info("Поиск: %r → %r%s", cmd.target_raw, target_en, " (по образцу)" if mine else "")
        dets = self.detector.detect(self.camera.read(), [target_en], conf=conf, pe=pe)
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

    def _count(self, cmd: Command) -> None:
        """«Сколько людей в комнате»: свежая детекция по одному классу.

        Без объекта («сколько всего предметов») считаются стандартные классы трекера.
        """
        lang = cmd.lang
        frame = self.camera.read()
        if cmd.target_en or cmd.target_raw:
            target_en = cmd.target_en or self._to_english(cmd.target_raw, lang)
            dets = self.detector.detect(frame, [target_en])
            self._last_dets = dets
            log.info("Подсчёт: %r → %r: %d", cmd.target_raw, target_en, len(dets))
            if not dets:
                obj = lexicon.object_name(target_en, lang) if cmd.target_en else cmd.target_raw
                self.tts.say(self.p("not_found", lang).format(
                    obj=obj if lang == "ru" else obj.capitalize()), lang, ANSWER)
                return
            items = lexicon.count_phrase(len(dets), cmd.target_en, lang, cmd.target_raw)
            self.tts.say(self.p("count", lang).format(items=items), lang, ANSWER)
            return
        dets = self.detector.detect(frame, self.cfg["detector"]["standard_classes"])
        self._last_dets = dets
        counts = Counter(d.name for d in dets)
        log.info("Подсчёт всего: %s", dict(counts))
        if not counts:
            self.tts.say(self.p("count_none", lang), lang, ANSWER)
            return
        items = ", ".join(lexicon.count_phrase(n, en, lang) for en, n in counts.most_common())
        self.tts.say(self.p("count_all", lang).format(items=items), lang, ANSWER)

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
        """«Запомни …»: лицо в кадре → человек, иначе (или по подсказке в команде) — предмет."""
        lang = cmd.lang
        if not cmd.name:
            self.tts.say(self.p("need_name", lang), lang, ANSWER)
            return
        frames = []
        for _ in range(self.cfg["faces" if self.faces else "objects"]["remember_samples"]):
            frames.append(self.camera.read())
            time.sleep(0.3)
        if cmd.obj is not True and self.faces:
            if self.faces.remember(frames, cmd.name):
                # человек, который сейчас ближе всех, теперь знакомый — не объявлять его заново
                people = [t for t in self.scene.tracks.values() if t.name == "person"]
                if people:
                    nearest = min(people, key=lambda t: t.det.meters or 99)
                    nearest.identity, nearest.announced = cmd.name, True
                self.tts.say(self.p("remembered", lang).format(name=cmd.name), lang, ANSWER)
                return
            if cmd.obj is False or self.objects is None:
                self.tts.say(self.p("no_face", lang), lang, ANSWER)
                return
        self._remember_object(cmd.name.lower(), frames, lang)

    def _remember_object(self, name: str, frames: list, lang: str) -> None:
        """Рамка предмета (YOLOE по названию) → визуальный эмбеддинг → база предметов.

        Предмет держат перед камерой, поэтому из нескольких рамок берём ближайшую
        к центру кадра. Не нашли ни одной — берём центральную область.
        """
        cls = lexicon.find_object(name) or self._to_english(name, lang)
        used, boxes = [], []
        for frame in frames:
            h, w = frame.shape[:2]
            dets = self.detector.detect(frame, [cls], conf=0.15)
            if dets:
                used.append(frame)
                boxes.append(max(dets, key=lambda d, w=w, h=h: d.conf
                                 - abs((d.box[0] + d.box[2]) / 2 / w - 0.5)
                                 - abs((d.box[1] + d.box[3]) / 2 / h - 0.5)).box)
        unsure = not used
        if unsure:
            h, w = frames[0].shape[:2]
            used, boxes = frames, [(w * 0.3, h * 0.25, w * 0.7, h * 0.85)] * len(frames)
        emb = self.detector.visual_embedding(used, boxes)
        self.objects.remember(name, cls, emb, len(used))
        log.info("Предмет %r (%s) запомнен по %d кадрам%s", name, cls, len(used), ", центр кадра" if unsure else "")
        self.tts.say(self.p("obj_unsure" if unsure else "remembered_obj", lang).format(name=name), lang, ANSWER)

    def _forget(self, cmd: Command) -> None:
        lang = cmd.lang
        removed = []
        # «забудь ключи» — предмет; «забудь всех» без подсказки трогает только людей
        if self.objects and cmd.obj is not False and (cmd.name or cmd.obj):
            removed += self.objects.forget(cmd.name or None)
        if self.faces and cmd.obj is not True and not removed:
            faces = self.faces.forget(cmd.name or None)
            for t in self.scene.tracks.values():
                if t.identity in faces:
                    t.identity = None
            removed += faces
        if not removed:
            self.tts.say(self.p("forgot_none", lang), lang, ANSWER)
            return
        self.tts.say(self.p("forgot", lang).format(names=", ".join(removed)), lang, ANSWER)

    def _memory(self, cmd: Command) -> None:
        lang = cmd.lang
        people = sorted(self.faces.db) if self.faces else []
        objects = sorted(self.objects.db) if self.objects else []
        if not people and not objects:
            self.tts.say(self.p("memory_empty", lang), lang, ANSWER)
            return
        none = self.p("none_word", lang)
        self.tts.say(self.p("memory_list", lang).format(
            people=", ".join(people) or none, objects=", ".join(objects) or none), lang, ANSWER)

    # --- фоновый трекинг -------------------------------------------------
    def _scan_loop(self) -> None:
        """Адаптивная частота трекинга, три режима:

        active — сцена меняется (новый/ушедший/приближающийся трек) или движется
                 весь кадр (человек идёт, поворачивает голову): scan_interval_s;
        watch  — что-то шевелится в части кадра: watch_interval_s;
        idle   — движения нет: контрольный прогон раз в idle_interval_s, но любое
                 движение (MotionGate) будит трекинг сразу.
        Без детектора движения idle не используется (только active и watch).
        """
        d = self.cfg["detector"]
        fast = d["scan_interval_s"]
        watch = d.get("watch_interval_s", 1.0)
        idle = d.get("idle_interval_s", 5.0)
        hold = d.get("active_hold_s", 3.0)
        m = self.motion
        last_change = time.time()  # первые секунды после запуска — частый режим
        stats = _ScanStats()
        while self._running:
            started = time.time()
            if m is not None:
                m.wake.clear()  # движение во время прогона разбудит следующий
            try:
                self._scan_once()
                if self.scene.changed:
                    last_change = started
            except Exception:
                log.exception("Ошибка фонового трекинга")
            now = time.time()
            stats.add(self.scan_mode, now - started)
            stats.maybe_log(now, m)

            if now - last_change < hold or (m is not None and now - m.last_global < hold):
                mode = "active"
            elif m is None or now - m.last_local < hold:
                mode = "watch"
            else:
                mode = "idle"
            if mode != self.scan_mode:
                log.debug("Фоновый трекинг: %s", mode)
                self.scan_mode = mode
            interval = {"active": fast, "watch": watch, "idle": idle}[mode]
            if mode == "watch" and self._busy.is_set():
                interval *= 2  # идёт команда (STT, VLM, перевод): меньше отнимаем у неё CPU
            left = interval - (now - started)
            if mode == "idle":
                m.wake.wait(max(0.0, left))
            else:
                time.sleep(max(0.0, left))

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
        near = t.det.distance != "far"
        if near and t.near_since is None:
            t.near_since = now
        # издалека лицо меньше MIN_FACE_PX: не тратим на него ни время, ни попытки
        if (self.faces and near and t.identity is None and t.face_tries < f["max_tries"]
                and now - t.last_face_try >= f["retry_interval_s"]):
            t.last_face_try = now
            x1, y1, x2, y2 = map(int, t.det.box)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size and is_sharp(crop):
                name, had_face = self.faces.identify(crop)
                t.face_tries += had_face  # человек спиной — попытка не сгорает
                if name:
                    t.identity = name
                    t.announced = True
                    self.tts.say(lexicon.sentence(self.p("known_person", lang).format(
                        name=name, **where)), lang, ANSWER)
                    return False
        if t.announced or t.identity:
            return False
        # незнакомец: объявляем, когда он рядом и лицо за отведённое время не узнали
        # пока лицо ни разу не нашли, «незнакомец» ничего не значит: ждём втрое дольше
        waited = (not self.faces or t.face_tries >= f["max_tries"]
                  or near and now - t.near_since >= f["unknown_after_s"] * (1 if t.face_tries else 3))
        if near and waited:
            t.announced = True
            return True
        return False


class _ScanStats:
    """Сводка фонового трекинга в лог раз в period_s: прогоны, время, режимы, CPU."""

    def __init__(self, period_s: float = 30.0):
        import os

        self.period_s = period_s
        self.cores = os.cpu_count() or 1
        self._reset(time.time())

    def _reset(self, now: float) -> None:
        self.t0, self.cpu0 = now, time.process_time()
        self.durations: list[float] = []
        self.modes: Counter[str] = Counter()

    def add(self, mode: str, duration: float) -> None:
        self.durations.append(duration)
        self.modes[mode] += 1

    @staticmethod
    def _motion_summary(motion) -> str:
        """Медиана, p90 и максимум доли движения за период — для подбора motion.local/global."""
        if not motion:
            return ""
        r = sorted(motion.drain())
        if not r:
            return ""
        return (f", движение: медиана {r[len(r) // 2]:.2%}, p90 {r[int(len(r) * 0.9)]:.2%}"
                f", макс {r[-1]:.2%}")

    def maybe_log(self, now: float, motion) -> None:
        wall = now - self.t0
        if wall < self.period_s or not self.durations:
            return
        ms = sorted(x * 1000 for x in self.durations)
        cpu = (time.process_time() - self.cpu0) / wall / self.cores * 100
        modes = ", ".join(f"{k} {v}" for k, v in self.modes.most_common())
        log.info("Трекинг за %.0f с: %d прогонов (%s), медиана %.0f мс, макс %.0f мс; "
                 "CPU процесса %.0f%%%s", wall, len(ms), modes, ms[len(ms) // 2], ms[-1], cpu,
                 self._motion_summary(motion))
        self._reset(now)
