from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["models_dir"] = str((ROOT / cfg.get("models_dir", "models")).resolve())
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    if "faces" in cfg:
        cfg["faces"]["db"] = str(ROOT / cfg["faces"]["db"])
    for section in ("vlm", "translator"):
        cfg[section]["path"] = str(Path(cfg["models_dir"]) / cfg[section]["path"])
    return cfg


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
