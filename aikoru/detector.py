"""Детекция объектов YOLOE с текстовыми запросами (open-vocabulary).

Расстояние оценивается грубо по одной камере: по известной средней высоте
объекта и высоте рамки в пикселях (модель камеры-обскуры). Для прототипа этого
достаточно; позже можно заменить на модель глубины.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

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


class Detector:
    def __init__(self, weights: str, device: str, conf: float = 0.3,
                 classes: list[str] | None = None, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLOE

        self.model = YOLOE(weights)
        self.device = device
        self.conf = conf
        self._lock = threading.Lock()
        self._pe_cache: dict[tuple[str, ...], object] = {}
        self._current: tuple[str, ...] = ()
        self.tracker = tracker
        if classes:
            self._set_classes(classes)

    def _set_classes(self, names: list[str]) -> None:
        key = tuple(names)
        if key == self._current:
            return
        if key not in self._pe_cache:
            self._pe_cache[key] = self.model.get_text_pe(list(key))
        self.model.set_classes(list(key), self._pe_cache[key])
        self.model.predictor = None  # иначе predictor продолжает работать со старыми классами
        self._current = key

    def detect(self, frame: np.ndarray, names: list[str], conf: float | None = None) -> list[Detection]:
        with self._lock:
            self._set_classes(names)
            res = self.model.predict(
                frame, conf=conf or self.conf, device=self.device, verbose=False
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
                device=self.device, verbose=False,
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
