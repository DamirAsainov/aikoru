"""Детекция объектов YOLOE с текстовыми запросами (open-vocabulary).

Расстояние оценивается грубо по одной камере: по известной средней высоте
объекта и высоте рамки в пикселях (модель камеры-обскуры). Для прототипа этого
достаточно; позже можно заменить на модель глубины.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

log = logging.getLogger(__name__)

# Средняя реальная высота объектов в метрах (для оценки расстояния)
REAL_HEIGHT_M = {
    "person": 1.7, "car": 1.5, "bus": 3.0, "truck": 3.0, "bicycle": 1.0,
    "motorcycle": 1.1, "door": 2.0, "traffic light": 0.9, "bench": 0.8,
    "chair": 0.9, "table": 0.75, "trash can": 0.9, "dog": 0.6, "cat": 0.3,
    "bottle": 0.25, "cup": 0.1, "backpack": 0.5, "laptop": 0.25,
}
CAMERA_HFOV_DEG = 70.0  # типичный угол обзора USB-вебкамеры


@dataclass
class Detection:
    name: str
    conf: float
    box: tuple[float, float, float, float]
    direction: str          # left | center | right
    distance: str           # very_close | close | far
    meters: float | None    # грубая оценка, если высота объекта известна
    track_id: int | None = None


class TextEmbeddings:
    """Дисковый кэш текстовых эмбеддингов YOLOE: слово → вектор.

    Без кэша YOLOE на каждый новый набор классов заново читает с диска
    текстовый энкодер MobileCLIP (572 МБ, ~0.7 с). Эмбеддинги слов независимы,
    поэтому энкодер нужен только для слов, которых ещё нет в кэше, и сразу
    выгружается — в памяти он не остаётся.
    """
    _shared: ClassVar[dict[str, TextEmbeddings]] = {}
    _shared_lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        if path.exists():
            with np.load(path) as z:
                self.vecs = {k: z[k] for k in z.files}

    @classmethod
    def for_weights(cls, weights: str) -> TextEmbeddings:
        """Один кэш на файл весов, общий для всех экземпляров Detector."""
        stem = Path(weights).stem.removesuffix("-seg")
        path = Path(weights).resolve().parent / f"{stem}.text_pe.npz"
        with cls._shared_lock:
            if str(path) not in cls._shared:
                cls._shared[str(path)] = cls(path)
            return cls._shared[str(path)]

    def get(self, model, names: list[str]):
        import torch

        with self._lock:
            missing = [n for n in dict.fromkeys(names) if n not in self.vecs]
            if missing:
                log.info("Кодирую текстовые запросы YOLOE: %s", ", ".join(missing))
                pe = model.get_text_pe(missing)[0].float().cpu().numpy()
                self.vecs.update(zip(missing, pe))
                tmp = self.path.with_suffix(".tmp.npz")
                np.savez(tmp, **self.vecs)
                tmp.replace(self.path)
            return torch.from_numpy(np.stack([self.vecs[n] for n in names]))[None]


def load_yoloe(weights: str):
    """YOLOE без ветки масок: маски не используются, а детекция так ~20% быстрее.

    Веса -seg.pt подходят целиком (голова детекции та же), Proto просто не строится.
    """
    from ultralytics import YOLOE

    if weights.endswith("-seg.pt"):
        # имя конфигурации без папки: ultralytics ищет yoloe-11s.yaml среди встроенных
        return YOLOE(Path(weights).name.replace("-seg.pt", ".yaml")).load(weights)
    return YOLOE(weights)


def openvino_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("openvino") is not None


def export_openvino(weights: str, classes: list[str], imgsz: tuple[int, int]) -> str:
    """YOLOE с зашитыми классами → модель OpenVINO (FP32, статичный вход h×w).

    Текстовые эмбеддинги впекаются в голову детекции, поэтому классы после
    экспорта не меняются: подходит только для трекера с постоянным набором.
    Экспорт делается один раз (~1 мин) и кладётся рядом с весами.
    """
    key = hashlib.sha1("|".join(classes).encode()).hexdigest()[:8]
    stem = Path(weights).stem.removesuffix("-seg")
    out = Path(weights).resolve().parent / f"{stem}-{key}-{imgsz[0]}x{imgsz[1]}_openvino_model"
    if (out / "metadata.yaml").exists():
        return str(out)
    log.info("Экспорт трекера в OpenVINO (%dx%d), один раз…", *imgsz)
    model = load_yoloe(weights)
    model.set_classes(list(classes), TextEmbeddings.for_weights(weights).get(model, list(classes)))
    tmp = Path(model.export(format="openvino", imgsz=list(imgsz), half=False, dynamic=False))
    tmp.replace(out)
    return str(out)


class Detector:
    def __init__(self, weights: str, device: str, conf: float = 0.3,
                 classes: list[str] | None = None, tracker: str = "bytetrack.yaml",
                 imgsz: int | tuple[int, int] = 640, backend: str = "torch"):
        """backend="openvino" — только для трекера: классы фиксируются при экспорте,
        detect() с другими классами и visual_embedding() недоступны."""
        self.backend = backend
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._pe_cache: dict[tuple[str, ...], object] = {}
        self._current: tuple[str, ...] = ()
        self._current_pe = None
        self._vp = None  # предиктор визуальных промптов, создаётся при первом запоминании
        self.tracker = tracker
        if backend == "openvino":
            from ultralytics import YOLO

            if not classes:
                raise ValueError("OpenVINO-детектору нужен постоянный набор классов")
            size = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
            self.model = YOLO(export_openvino(weights, classes, size), task="detect")
            self._current = tuple(classes)
            return
        self.model = load_yoloe(weights)
        self.text = TextEmbeddings.for_weights(weights)
        if classes:
            self._set_classes(classes)

    def _set_classes(self, names: list[str], pe=None) -> None:
        """pe — готовые эмбеддинги (например, визуальные); без них берутся текстовые."""
        key = tuple(names)
        if self.backend == "openvino":
            if key != self._current or pe is not None:
                raise ValueError("Классы OpenVINO-модели зафиксированы при экспорте")
            return
        if pe is None:
            if key not in self._pe_cache:
                self._pe_cache[key] = self.text.get(self.model, list(key))
            pe = self._pe_cache[key]
        if key == self._current and pe is self._current_pe:
            return
        self.model.model.set_classes(list(key), pe)
        if not self._swap_predictor_classes(list(key), pe):
            self.model.predictor = None  # иначе predictor продолжает работать со старыми классами
        self._current, self._current_pe = key, pe

    def _swap_predictor_classes(self, names: list[str], pe) -> bool:
        """Меняет классы прямо в копии модели внутри predictor.

        Пересоздание predictor (deepcopy + fuse) стоит ~0.3 с на каждую смену
        запроса, подмена эмбеддингов — миллисекунды.
        """
        try:
            backend = self.model.predictor.model.backend
            device = next(backend.model.parameters()).device
            backend.model.set_classes(names, pe.to(device))
            backend.names = dict(enumerate(names))
            return True
        except AttributeError:  # predictor ещё не создан или другой бэкенд
            return False

    def visual_embedding(self, frames: list[np.ndarray], boxes: list[tuple]) -> np.ndarray:
        """Визуальный промпт YOLOE: рамки объекта на кадрах → один нормированный эмбеддинг.

        Считается только backbone + SAVPE (без головы детекции), ~20 мс на кадр.
        """
        import torch

        with self._lock:
            if self._vp is None:
                from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

                self._vp = YOLOEVPDetectPredictor(overrides={
                    "task": "detect", "mode": "predict", "save": False, "verbose": False,
                    "batch": 1, "device": self.device, "imgsz": self.imgsz,
                })
                self._vp.setup_model(self.model.model, verbose=False)
            embs = []
            for frame, box in zip(frames, boxes):
                self._vp.set_prompts({"bboxes": np.array([box], dtype=np.float32), "cls": np.array([0])})
                embs.append(self._vp.get_vpe(frame)[0, 0].float())
            emb = torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=0)
        return emb.cpu().numpy()

    def detect(self, frame: np.ndarray, names: list[str], conf: float | None = None,
               pe=None) -> list[Detection]:
        with self._lock:
            self._set_classes(names, pe)
            res = self.model.predict(
                frame, conf=conf or self.conf, device=self.device, imgsz=self.imgsz, verbose=False
            )[0]
        return self._parse(res, list(names), frame.shape)

    def track(self, frame: np.ndarray, conf: float = 0.1) -> list[Detection]:
        """Детекция + трекинг (ByteTrack) по классам, заданным в конструкторе.

        Низкий conf нужен ByteTrack: слабые детекции помогают не терять треки.
        Возвращаются только объекты с номером трека.
        """
        with self._lock:
            res = self.model.track(
                frame, persist=True, tracker=self.tracker, conf=conf,
                device=self.device, imgsz=self.imgsz, verbose=False,
            )[0]
        dets = self._parse(res, list(self._current), frame.shape)
        return [d for d in dets if d.track_id is not None]

    def _parse(self, res, names: list[str], shape) -> list[Detection]:
        h, w = shape[:2]
        boxes = res.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        out = []
        for box, c, cls, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), ids):
            det = self._describe(names[int(cls)], float(c), tuple(box), w, h)
            det.track_id = tid
            out.append(det)
        out.sort(key=lambda d: (d.meters if d.meters is not None else 99, -d.conf))
        return out

    @staticmethod
    def _describe(name, conf, box, w, h) -> Detection:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / w
        direction = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"

        box_h = max(y2 - y1, 1.0)
        meters = None
        if name in REAL_HEIGHT_M:
            # Высота кадра → фокус в пикселях (вертикальный FOV из горизонтального)
            f_px = (w / 2) / math.tan(math.radians(CAMERA_HFOV_DEG / 2))
            meters = REAL_HEIGHT_M[name] * f_px / box_h
            # Рамка обрезана краем кадра — объект ближе, чем видно
            if y1 <= 2 or y2 >= h - 2:
                meters *= 0.7
            meters = round(meters, 1)

        if meters is not None:
            distance = "very_close" if meters < 1.5 else "close" if meters < 4 else "far"
        else:
            ratio = box_h / h
            distance = "very_close" if ratio > 0.6 else "close" if ratio > 0.3 else "far"
        return Detection(name, conf, box, direction, distance, meters)
