"""Дешёвый детектор движения (MOG2) перед YOLOE: решает, как часто гонять трекинг.

Работает в потоке камеры на сером кадре 160×120 (~1 мс). Камера носится на
человеке, поэтому доля «движущихся» пикселей отвечает на два вопроса:
  - local  — что-то шевелится в кадре (вошёл человек, едет машина);
  - global — меняется почти весь кадр: человек идёт или поворачивает голову
             (или резко сменилось освещение — тогда лишний частый прогон не страшен).
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class MotionGate:
    def __init__(self, width: int = 160, local: float = 0.003, global_: float = 0.15,
                 history: int = 300, var_threshold: float = 25, min_interval_s: float = 0.07):
        self.width = width
        self.local = local
        self.global_ = global_
        self.min_interval_s = min_interval_s
        self._mog = cv2.createBackgroundSubtractorMOG2(history, var_threshold, detectShadows=False)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._last_feed = 0.0
        self._warmup = 10           # первые кадры MOG2 ещё строит фон
        self.ratio = 0.0            # доля движущихся пикселей в последнем кадре
        self.last_local = 0.0       # time.time() последнего локального движения
        self.last_global = 0.0      # time.time() последнего движения всего кадра
        self.wake = threading.Event()  # будит трекинг из спокойного режима

    def feed(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self._last_feed < self.min_interval_s:
            return
        self._last_feed = now
        h = round(frame.shape[0] * self.width / frame.shape[1])
        small = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = cv2.morphologyEx(self._mog.apply(gray), cv2.MORPH_OPEN, self._kernel)
        if self._warmup:
            self._warmup -= 1
            return
        self.ratio = cv2.countNonZero(mask) / mask.size
        if self.ratio >= self.global_:
            self.last_global = now
        if self.ratio >= self.local:
            self.last_local = now
            self.wake.set()
