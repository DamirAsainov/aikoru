"""Распознавание знакомых лиц: OpenCV YuNet (детекция) + SFace (эмбеддинг 128-d).

База хранится локально в data/faces.json: {имя: [эмбеддинг, ...]}.
Это биометрические данные — файл не покидает устройство и не попадает в git.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
FACE_URLS = {
    YUNET: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET,
    SFACE: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/" + SFACE,
}
MIN_FACE_PX = 40          # меньшие лица распознаются ненадёжно
MAX_SAMPLES_PER_NAME = 10


class FaceBook:
    def __init__(self, models_dir: str, db_path: str, threshold: float = 0.40):
        d = Path(models_dir) / "faces"
        for f in (YUNET, SFACE):
            if not (d / f).exists():
                raise FileNotFoundError(f"{d / f} не найден — запустите scripts/download_models.py")
        self.detector = cv2.FaceDetectorYN.create(str(d / YUNET), "", (320, 320), 0.8)
        self.recognizer = cv2.FaceRecognizerSF.create(str(d / SFACE), "")
        self.threshold = threshold  # косинусное сходство SFace; рекомендуемый порог 0.363
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.db: dict[str, list[np.ndarray]] = {}
        if self.db_path.exists():
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
            self.db = {name: [np.array(e, dtype=np.float32) for e in embs] for name, embs in raw.items()}
        log.info("Знакомых лиц в базе: %d (%s)", len(self.db), ", ".join(self.db) or "—")

    # --- низкий уровень --------------------------------------------------
    def _faces(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        with self._lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
        if faces is None:
            return np.zeros((0, 15), dtype=np.float32)
        faces = faces[faces[:, 2] >= MIN_FACE_PX]
        return faces[np.argsort(-faces[:, 2] * faces[:, 3])]  # крупные первыми

    def _embed(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        with self._lock:
            aligned = self.recognizer.alignCrop(image, face)
            emb = self.recognizer.feature(aligned).flatten()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _match(self, emb: np.ndarray) -> tuple[str | None, float]:
        best, best_score = None, -1.0
        for name, embs in self.db.items():
            score = max(float(np.dot(emb, e)) for e in embs)
            if score > best_score:
                best, best_score = name, score
        return (best, best_score) if best_score >= self.threshold else (None, best_score)

    # --- API --------------------------------------------------------------
    def identify(self, image: np.ndarray) -> tuple[str | None, bool]:
        """Самое крупное лицо на изображении → (имя или None, было ли лицо вообще)."""
        faces = self._faces(image)
        if len(faces) == 0:
            return None, False
        name, score = self._match(self._embed(image, faces[0]))
        log.debug("Лицо: %s (%.2f)", name, score)
        return name, True

    def identify_all(self, frame: np.ndarray) -> list[tuple[str | None, float]]:
        """Все лица в кадре → [(имя или None, центр x 0..1)]."""
        w = frame.shape[1]
        out = []
        for face in self._faces(frame):
            name, _ = self._match(self._embed(frame, face))
            out.append((name, float(face[0] + face[2] / 2) / w))
        return out

    def remember(self, frames: list[np.ndarray], name: str) -> int:
        """Сохраняет самое крупное лицо из каждого кадра под именем. Возвращает число образцов."""
        embs = []
        for frame in frames:
            faces = self._faces(frame)
            if len(faces):
                embs.append(self._embed(frame, faces[0]))
        if not embs:
            return 0
        with self._lock:
            self.db[name] = (self.db.get(name, []) + embs)[-MAX_SAMPLES_PER_NAME:]
            self._save()
        return len(embs)

    def forget(self, name: str | None) -> list[str]:
        """Удаляет человека (нечёткое совпадение имени) или всех, если name is None."""
        with self._lock:
            if name is None:
                removed = list(self.db)
            else:
                key = name.lower().replace("ё", "е")
                removed = [n for n in self.db if n.lower().replace("ё", "е").startswith(key[:4])]
            for n in removed:
                del self.db[n]
            self._save()
        return removed

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {n: [e.round(5).tolist() for e in embs] for n, embs in self.db.items()}
        self.db_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
