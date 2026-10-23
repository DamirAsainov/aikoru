"""Описание сцены и ответы на вопросы о кадре: малая VLM через transformers.

По умолчанию LFM2.5-VL-450M (SigLIP2 + LFM2-350M). Подходит любая модель
с AutoModelForImageTextToText и chat-шаблоном, например SmolVLM-256M-Instruct.

Скорость на слабом CPU определяется числом токенов изображения: кадр без
нарезки на тайлы и 64 токена дают ~1.5–3 с на ответ вместо ~5 с (256 токенов).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PROMPTS = {
    "describe": "Describe the scene in one short sentence for a blind person: "
                "main objects and obstacles, where they are.",
    "ask": "{q} Answer briefly, in one sentence.",
}


def ensure_tokenizer_compat(model_dir: str | Path) -> None:
    """LFM2.5-VL сохранён в формате transformers 5; для 4.x правим tokenizer_config.json."""
    path = Path(model_dir) / "tokenizer_config.json"
    if not path.exists():
        return
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("tokenizer_class") != "TokenizersBackend":
        return
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("backend", None)
    special = cfg.get("model_specific_special_tokens") or {}
    cfg["extra_special_tokens"] = {**special, **({"image_thumbnail": cfg["image_thumbnail"]}
                                                 if "image_thumbnail" in cfg else {})}
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("tokenizer_config.json приведён к формату transformers 4.x")

# llama.cpp режет крупный кадр LFM2-VL на тайлы 512 px (7 тайлов на 640×480 — ~25 с на CPU),
# флага, чтобы отключить нарезку, нет. Кадр до 384 px кодируется целиком за ~0.4 с.
LLAMA_MAX_SIDE = 384


def trim_answer(answer: str) -> str:
    """Ответ обрезан лимитом токенов — оставляем только законченные предложения."""
    answer = answer.strip()
    if "." in answer and not answer.endswith((".", "!", "?")):
        answer = answer[:answer.rfind(".") + 1]
    return answer


class SceneDescriber:
    def __init__(self, model_dir: str, device: str, image_tokens: int = 64, max_new_tokens: int = 40,
                 threads: int = 0):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # «layer_norm newly initialized» — слой ниже отключается
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} не найден — запустите scripts/download_models.py")
        ensure_tokenizer_compat(model_dir)
        # fp16 только на GPU с тензорными ядрами; на CPU bf16/fp16 в разы медленнее fp32
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        proj = getattr(self.model.model, "multi_modal_projector", None)
        if getattr(self.model.config, "projector_use_layernorm", True) is False and proj is not None:
            # transformers 4.x всегда ставит LayerNorm в проектор LFM2-VL; LFM2.5 обучена без
            # него, и случайно инициализированный слой искажает признаки изображения
            proj.layer_norm = torch.nn.Identity()
        kw = {"do_image_splitting": False}
        if self.model.config.model_type == "lfm2_vl":
            kw.update(max_image_tokens=image_tokens, min_image_tokens=min(image_tokens, 64))
        self.proc = AutoProcessor.from_pretrained(model_dir, **kw)
        self.device = device
        self.max_new_tokens = max_new_tokens
        # половина ядер: трекер во время ответа VLM не должен остаться без CPU
        self.threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self._lock = threading.Lock()

    def _generate(self, frame: np.ndarray, text: str) -> str:
        import torch

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(images=[image], text=prompt, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        n = inputs["input_ids"].shape[1]
        with self._lock, torch.inference_mode():
            prev = torch.get_num_threads()
            if self.device == "cpu":
                torch.set_num_threads(self.threads)
            try:
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            finally:
                torch.set_num_threads(prev)
        return trim_answer(self.proc.batch_decode(out[:, n:], skip_special_tokens=True)[0])

    def describe(self, frame: np.ndarray) -> str:
        """Короткое описание сцены на английском."""
        return self._generate(frame, PROMPTS["describe"])

    def ask(self, frame: np.ndarray, question_en: str) -> str:
        return self._generate(frame, PROMPTS["ask"].format(q=question_en))


class LlamaCppDescriber:
    """Та же VLM в GGUF через llama-server (llama.cpp) — отдельный процесс на localhost.

    Квантованная модель (Q8_0 + mmproj) занимает ~0.5 ГБ вместо ~1.7 ГБ fp32 в torch,
    а llama.cpp на CPU заметно быстрее transformers. Сервер не открывает порт наружу.
    """

    def __init__(self, server: str, model: str, mmproj: str, image_tokens: int = 64,
                 max_new_tokens: int = 40, threads: int = 0, port: int = 8766):
        import atexit
        import subprocess
        import time
        import urllib.request

        for f in (server, model, mmproj):
            if not Path(f).exists():
                raise FileNotFoundError(f"{f} не найден — запустите scripts/download_models.py")
        self.max_new_tokens = max_new_tokens
        self.url = f"http://127.0.0.1:{port}"
        threads = threads or max(1, (os.cpu_count() or 2) // 2)
        cmd = [server, "-m", model, "--mmproj", mmproj, "--host", "127.0.0.1", "--port", str(port),
               "-t", str(threads), "-tb", str(threads), "-c", "2048", "-np", "1", "-ngl", "0",
               "--image-min-tokens", str(min(image_tokens, 64)), "--image-max-tokens", str(image_tokens),
               "--no-webui", "--log-disable"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=flags)
        atexit.register(self.close)
        self._lock = threading.Lock()
        deadline = time.time() + 120
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server завершился с кодом {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.time() > deadline:
                self.close()
                raise RuntimeError("llama-server не запустился за 120 с")
            time.sleep(0.2)

    def _generate(self, frame: np.ndarray, text: str) -> str:
        import base64
        import urllib.request

        h, w = frame.shape[:2]
        scale = LLAMA_MAX_SIDE / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Не удалось закодировать кадр")
        body = json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}},
                {"type": "text", "text": text}]}],
            "max_tokens": self.max_new_tokens, "temperature": 0, "cache_prompt": False,
        }).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with self._lock, urllib.request.urlopen(req, timeout=60) as r:
            answer = json.load(r)["choices"][0]["message"]["content"]
        return trim_answer(answer)

    def describe(self, frame: np.ndarray) -> str:
        return self._generate(frame, PROMPTS["describe"])

    def ask(self, frame: np.ndarray, question_en: str) -> str:
        return self._generate(frame, PROMPTS["ask"].format(q=question_en))

    def close(self) -> None:
        import subprocess

        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
