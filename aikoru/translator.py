"""Офлайн-перевод NLLB-200 (CTranslate2 int8) между en, ru и kk.

VLM и текстовые запросы YOLOE работают только на английском.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from pathlib import Path

NLLB_CODES = {"en": "eng_Latn", "ru": "rus_Cyrl", "kk": "kaz_Cyrl"}


class Translator:
    def __init__(self, model_dir: str, device: str, threads: int = 0):
        import ctranslate2
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # ложное предупреждение о regex токенизатора

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} не найден — запустите scripts/download_models.py")
        compute = "int8_float16" if device == "cuda" else "int8"
        self.model = ctranslate2.Translator(model_dir, device=device, compute_type=compute,
                                           intra_threads=threads)  # 0 — по умолчанию CTranslate2
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()  # фразы повторяются

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        key = (src, tgt, text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            self.tok.src_lang = NLLB_CODES[src]
            tokens = self.tok.convert_ids_to_tokens(self.tok.encode(text))
            res = self.model.translate_batch(
                [tokens], target_prefix=[[NLLB_CODES[tgt]]], beam_size=2, max_decoding_length=256
            )
            out = res[0].hypotheses[0][1:]  # без кода языка
            text = self.tok.decode(self.tok.convert_tokens_to_ids(out), skip_special_tokens=True)
            text = re.sub(r"\s+([.,!?;:'])", r"\1", text).strip()
            self._cache[key] = text
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
            return text
