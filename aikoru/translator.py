"""Офлайн-перевод NLLB-200 (CTranslate2 int8) между en, ru и kk.

Moondream и текстовые запросы YOLOE работают только на английском.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

NLLB_CODES = {"en": "eng_Latn", "ru": "rus_Cyrl", "kk": "kaz_Cyrl"}


class Translator:
    def __init__(self, model_dir: str, device: str):
        import ctranslate2
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # ложное предупреждение о regex токенизатора

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} не найден — запустите scripts/download_models.py")
        compute = "int8_float16" if device == "cuda" else "int8"
        self.model = ctranslate2.Translator(model_dir, device=device, compute_type=compute)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self._lock = threading.Lock()

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        with self._lock:
            self.tok.src_lang = NLLB_CODES[src]
            tokens = self.tok.convert_ids_to_tokens(self.tok.encode(text))
            res = self.model.translate_batch(
                [tokens], target_prefix=[[NLLB_CODES[tgt]]], beam_size=2, max_decoding_length=256
            )
            out = res[0].hypotheses[0][1:]  # без кода языка
            text = self.tok.decode(self.tok.convert_tokens_to_ids(out), skip_special_tokens=True)
            return re.sub(r"\s+([.,!?;:'])", r"\1", text).strip()
