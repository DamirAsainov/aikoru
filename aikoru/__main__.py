"""Запуск: python -m aikoru [--text] [--show] [--config path]"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="aikoru", description="AIKORU — офлайн-ассистент ориентации")
    ap.add_argument("--config", help="путь к config.yaml")
    ap.add_argument("--text", action="store_true", help="команды с клавиатуры вместо микрофона")
    ap.add_argument("--show", action="store_true", help="окно с видео и рамками (Esc — выход)")
    ap.add_argument("--online", action="store_true", help="разрешить загрузку моделей из сети")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from .config import load_config

    cfg = load_config(args.config)
    prepare_env(cfg["models_dir"], offline=not args.online)

    from .assistant import Assistant

    try:
        Assistant(cfg, voice_input=not args.text, show=args.show).run()
    except KeyboardInterrupt:
        pass


def prepare_env(models_dir: str, offline: bool) -> None:
    """Все модели хранятся в models/; без --online сеть не используется."""
    os.makedirs(models_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(models_dir, "hf"))
    os.environ.setdefault("TORCH_HOME", os.path.join(models_dir, "torch"))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["YOLO_OFFLINE"] = "1"
    # ultralytics кладёт веса YOLOE и текстовый энкодер MobileCLIP в текущую папку
    os.chdir(models_dir)


if __name__ == "__main__":
    main()
