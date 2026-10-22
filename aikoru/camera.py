"""Фоновый захват кадров с USB-камеры: всегда отдаёт самый свежий кадр."""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 15):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру #{index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 30 кадров/с не нужны: трекинг не чаще ~3 раз/с, MOG2 ~14 раз/с, веб ~12 раз/с
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self.motion = None  # MotionGate: получает каждый кадр в этом потоке
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                log.warning("Камера не вернула кадр")
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame
            if self.motion is not None:
                self.motion.feed(frame)

    def read(self, timeout: float = 3.0) -> np.ndarray:
        """Последний кадр в BGR."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            if time.time() > deadline:
                raise RuntimeError("Нет кадров с камеры")
            time.sleep(0.02)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()
