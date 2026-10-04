"""Однократная загрузка всех моделей в папку models/. После этого AIKORU работает офлайн.

    python scripts/download_models.py
"""
from __future__ import annotations

import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aikoru.__main__ import prepare_env  # noqa: E402
from aikoru.config import load_config  # noqa: E402
from aikoru.tts import SILERO_URL, tts_model_path  # noqa: E402
from scripts.hf_fetch import fetch_repo  # noqa: E402

VOSK_URL = "https://alphacephei.com/vosk/models/{name}.zip"
log = logging.getLogger("download")


def fetch(url: str) -> bytes:
    log.info("Скачиваю %s", url)
    with urllib.request.urlopen(url) as r:
        return r.read()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    models = Path(cfg["models_dir"])
    prepare_env(str(models), offline=False)

    # 1. Silero TTS
    t = cfg["tts"]
    path = tts_model_path(str(models), t["model"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(SILERO_URL.format(model=t["model"])))

    # 2. Vosk ru + kk
    for name in cfg["voice"]["vosk_models"].values():
        if not (models / "vosk" / name).exists():
            zipfile.ZipFile(io.BytesIO(fetch(VOSK_URL.format(name=name)))).extractall(models / "vosk")

    # 3. Silero VAD входит в pip-пакет silero-vad — скачивать не нужно.

    # 3b. Распознавание лиц: OpenCV YuNet + SFace
    from aikoru.faces import FACE_URLS

    for name, url in FACE_URLS.items():
        out = models / "faces" / name
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(fetch(url))

    # 4. Moondream2 и NLLB (CTranslate2 int8) — в локальные папки models/
    v = cfg["vlm"]
    fetch_repo(v["repo"], v["revision"], v["path"], exclude=[".gguf", ".png", ".jpg", ".md"])
    # код Moondream грузит токенизатор из отдельного репозитория через HF-кэш (models/hf)
    from huggingface_hub import hf_hub_download

    hf_hub_download("moondream/starmie-v1", "tokenizer.json")
    t = cfg["translator"]
    fetch_repo(t["repo"], "main", t["path"], exclude=[".md"])

    # 5. YOLOE + текстовый энкодер MobileCLIP (ultralytics качает сам)
    from aikoru.detector import Detector

    det = Detector(cfg["detector"]["weights"], cfg["device"])
    det.model.get_text_pe(["door"])
    log.info("YOLOE готов")
    log.info("Все модели загружены в %s", models)


if __name__ == "__main__":
    main()
