"""Описание сцены через SmolVLM-500M (локально, по голосовой команде)."""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

CAPTION_PROMPTS = {
    "short": "Describe this image in one short sentence.",
    "normal": "Describe this image in two or three sentences.",
}
# Камера низкого разрешения, а число патчей SmolVLM растёт с размером кадра:
# ограничиваем длинную сторону, чтобы описание укладывалось в 2 ГБ видеопамяти
MAX_SIDE = 512


class SceneDescriber:
    def __init__(self, model_dir: str, device: str, caption_length: str = "short"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # fp16 — только на GPU; bf16 на CPU вдвое экономнее fp32
        self.dtype = torch.float16 if device == "cuda" else torch.bfloat16
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_dir, size={"longest_edge": MAX_SIDE})
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=self.dtype).to(device).eval()
        self.caption_length = caption_length
        self._lock = threading.Lock()

    @staticmethod
    def _to_pil(frame: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _generate(self, frame: np.ndarray, prompt: str, max_new_tokens: int) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[self._to_pil(frame)], return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with self._lock, torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()

    def describe(self, frame: np.ndarray) -> str:
        """Короткое описание сцены на английском."""
        return self._generate(frame, CAPTION_PROMPTS.get(self.caption_length, CAPTION_PROMPTS["short"]), 60)

    def ask(self, frame: np.ndarray, question_en: str) -> str:
        return self._generate(frame, question_en, 80)
