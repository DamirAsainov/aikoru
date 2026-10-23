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
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-{asset}"
log = logging.getLogger("download")


def fetch(url: str) -> bytes:
    log.info("Скачиваю %s", url)
    with urllib.request.urlopen(url) as r:
        return r.read()


def fetch_llama_cpp(v: dict, models: Path) -> None:
    """llama-server (CPU-сборка) и GGUF-файлы VLM для backend: llamacpp."""
    import os
    import tarfile

    server = Path(v["server"])
    if not server.exists() and v.get("llama_cpp_release"):
        asset = "win-cpu-x64.zip" if os.name == "nt" else "ubuntu-x64.tar.gz"
        data = fetch(LLAMA_CPP_URL.format(tag=v["llama_cpp_release"], asset=asset))
        server.parent.mkdir(parents=True, exist_ok=True)
        if asset.endswith(".zip"):
            zipfile.ZipFile(io.BytesIO(data)).extractall(server.parent)
        else:  # в архиве папка llama-<tag>/: кладём содержимое прямо в models/llama.cpp
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                for m in tar.getmembers():
                    m.name = m.name.split("/", 1)[-1]
                    if m.name and (m.isfile() or m.issym()):
                        tar.extract(m, server.parent, filter="data")
    for key in ("gguf", "mmproj"):
        out = Path(v[key]) if v.get(key) else None
        if out and not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(fetch(f"https://huggingface.co/{v['gguf_repo']}/resolve/main/{out.name}"))
            tmp.replace(out)


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

    # 4. VLM и NLLB (CTranslate2 int8) — в локальные папки models/
    from aikoru.vlm import ensure_tokenizer_compat

    v = cfg["vlm"]
    fetch_repo(v["repo"], v["revision"], v["path"], exclude=[".gguf", ".png", ".jpg", ".md", "onnx/"])
    ensure_tokenizer_compat(v["path"])
    fetch_llama_cpp(v, models)
    t = cfg["translator"]
    fetch_repo(t["repo"], "main", t["path"], exclude=[".md"])

    # 5. YOLOE + текстовый энкодер MobileCLIP (ultralytics качает сам)
    from aikoru.detector import Detector

    # Эмбеддинги всех классов словаря считаются один раз и кэшируются на диске:
    # во время работы текстовый энкодер (572 МБ) нужен только для новых слов
    from aikoru.lexicon import OBJECT_STEMS

    det = Detector(cfg["detector"]["weights"], "cpu")
    names = sorted(set(cfg["detector"]["standard_classes"]) | set(OBJECT_STEMS.values()))
    det.text.get(det.model, names)
    log.info("YOLOE готов, текстовых запросов в кэше: %d", len(det.text.vecs))
    log.info("Все модели загружены в %s", models)


if __name__ == "__main__":
    main()
