"""Описание сцены через Moondream2 (локально, по голосовой команде)."""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _prime_module_cache(model_dir: str) -> None:
    """transformers 4.x копирует из локальной папки не все вложенные импорты
    удалённого кода Moondream (layers.py и др.) — копируем все .py сами."""
    import shutil
    from pathlib import Path

    from transformers.utils import HF_MODULES_CACHE

    src = Path(model_dir)
    dst = Path(HF_MODULES_CACHE) / "transformers_modules" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "__init__.py").touch()
    for f in src.glob("*.py"):
        shutil.copy2(f, dst / f.name)


class SceneDescriber:
    def __init__(self, model_dir: str, device: str, caption_length: str = "short"):
        import torch
        from transformers import AutoModelForCausalLM

        _prime_module_cache(model_dir)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)
        self.caption_length = caption_length
        self._lock = threading.Lock()

    @staticmethod
    def _to_pil(frame: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def describe(self, frame: np.ndarray) -> str:
        """Короткое описание сцены на английском."""
        with self._lock:
            return self.model.caption(self._to_pil(frame), length=self.caption_length)["caption"].strip()

    def ask(self, frame: np.ndarray, question_en: str) -> str:
        with self._lock:
            return self.model.query(self._to_pil(frame), question_en)["answer"].strip()
