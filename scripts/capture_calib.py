"""Кадры с камеры для INT8-калибровки трекера (detector.int8: true).

    python scripts/capture_calib.py [--n 300] [--every 0.5]

Носите камеру как обычно: комната, коридор, улица, люди, двери. Кадры кладутся
в detector.calib_dir (data/calib, не в git). После этого при запуске
AIKORU один раз экспортирует INT8-модель (~1–3 мин). Чтобы откалибровать
заново, удалите папку yoloe-…-int8_openvino_model рядом с весами.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from aikoru.camera import Camera  # noqa: E402
from aikoru.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="сколько кадров снять")
    ap.add_argument("--every", type=float, default=0.5, help="секунд между кадрами")
    args = ap.parse_args()

    cfg = load_config()
    out = Path(cfg["detector"]["calib_dir"])
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["camera"]
    cam = Camera(c["index"], c["width"], c["height"], c.get("fps", 15))
    start = len(list(out.glob("*.jpg")))
    try:
        for i in range(start, start + args.n):
            cv2.imwrite(str(out / f"{i:05d}.jpg"), cam.read(), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"\r{i - start + 1}/{args.n}", end="", flush=True)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
    print(f"\nКадров в {out}: {len(list(out.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
