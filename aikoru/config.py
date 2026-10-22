from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["models_dir"] = str((ROOT / cfg.get("models_dir", "models")).resolve())
    prepare_openmp(cfg)  # до первого импорта torch
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    for section in ("faces", "objects"):
        if section in cfg:
            cfg[section]["db"] = str(ROOT / cfg[section]["db"])
    for section in ("vlm", "translator"):
        cfg[section]["path"] = str(Path(cfg["models_dir"]) / cfg[section]["path"])
        # тяжёлым моделям GPU нужен только при достаточном запасе VRAM, иначе — CPU
        cfg[section]["device"] = resolve_device(cfg[section].get("device", "auto"),
                                                cfg[section].get("min_vram_gb", 0))
    return cfg


def prepare_openmp(cfg: dict) -> None:
    """Включает OMP_WAIT_POLICY=PASSIVE, если трекер пойдёт на CPU через OpenVINO.

    Иначе потоки OpenMP у torch после каждой операции крутятся в ожидании и
    отнимают ядра у OpenVINO: на i5-10400 трекер 150 мс вместо 50 мс. Цена —
    NLLB на ~0.13 с медленнее на фразу, поэтому с видеокартой не включается.
    Действует только до импорта torch: решение принимается без него, по драйверу CUDA.
    """
    import ctypes.util
    import importlib.util
    import os

    backend = cfg.get("detector", {}).get("backend", "auto")
    if backend == "torch" or importlib.util.find_spec("openvino") is None:
        return
    cuda_driver = ctypes.util.find_library("nvcuda" if os.name == "nt" else "cuda")
    if backend == "openvino" or cfg.get("device", "auto") == "cpu" or not cuda_driver:
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def openmp_passive() -> bool:
    import os

    return os.environ.get("OMP_WAIT_POLICY", "").upper() == "PASSIVE"


def resolve_device(device: str, min_vram_gb: float = 0) -> str:
    if device != "auto":
        return device
    import torch

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return "cpu"
        major, minor = torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError) as e:  # драйвер есть, а устройства не видно
        log.warning("CUDA недоступна (%s), работаю на CPU", e)
        return "cpu"
    archs = torch.cuda.get_arch_list()
    ptx = [int(a.split("_")[1]) for a in archs if a.startswith("compute_")]
    if f"sm_{major}{minor}" not in archs and not any(p <= major * 10 + minor for p in ptx):
        # напр. MX350 (sm_61) и колёса torch cu128: нужен torch для CUDA 12.6, см. README
        log.warning("Сборка torch не поддерживает %s (sm_%d%d), работаю на CPU",
                    torch.cuda.get_device_name(), major, minor)
        return "cpu"
    if torch.cuda.get_device_properties(0).total_memory < min_vram_gb * 2**30:
        return "cpu"
    return "cuda"
