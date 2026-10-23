import time
from types import SimpleNamespace

import numpy as np

from aikoru.assistant import Assistant, _ScanStats
from aikoru.faces import is_sharp
from aikoru.motion import MotionGate
from aikoru.translator import Translator


def test_is_sharp_rejects_flat_and_accepts_texture():
    flat = np.full((64, 64, 3), 128, np.uint8)
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert not is_sharp(flat)
    assert is_sharp(noise)


def test_motion_drain_clears():
    m = MotionGate()
    m._ratios.extend([0.1, 0.2])
    assert m.drain() == [0.1, 0.2]
    assert m.drain() == []


def test_motion_summary_percentiles():
    m = SimpleNamespace(drain=lambda: [i / 100 for i in range(100)])
    s = _ScanStats._motion_summary(m)
    assert "медиана 50.00%" in s and "p90 90.00%" in s and "макс 99.00%" in s
    assert _ScanStats._motion_summary(None) == ""
    assert _ScanStats._motion_summary(SimpleNamespace(drain=list)) == ""


def test_translator_cache_hits_and_evicts():
    calls = []

    class Fake(Translator):
        def __init__(self):
            import threading
            from collections import OrderedDict
            self._lock, self._cache = threading.Lock(), OrderedDict()
            self.tok = SimpleNamespace(
                encode=lambda t: calls.append(t) or [1],
                convert_ids_to_tokens=lambda ids: ["x"],
                convert_tokens_to_ids=lambda t: [1],
                decode=lambda ids, skip_special_tokens: "привет ,",
            )
            self.model = SimpleNamespace(translate_batch=lambda *a, **k: [
                SimpleNamespace(hypotheses=[["code", "y"]])])

    t = Fake()
    assert t.translate("hello", "en", "ru") == "привет,"
    assert t.translate("hello", "en", "ru") == "привет,"
    assert len(calls) == 1
    for i in range(300):
        t.translate(f"w{i}", "en", "ru")
    assert len(t._cache) == 256


def _static(motion, scene_changed=False):
    return SimpleNamespace(motion=motion, scene=SimpleNamespace(changed=scene_changed))


def test_scene_static_since():
    check = Assistant._scene_static_since
    now = time.time()
    assert check(_static(SimpleNamespace(last_local=now - 10)), now - 5)
    assert not check(_static(SimpleNamespace(last_local=now - 1)), now - 5)       # было движение
    assert not check(_static(SimpleNamespace(last_local=0), True), now - 5)       # сцена изменилась
    assert not check(_static(SimpleNamespace(last_local=0)), now - 500)           # устарело
    assert not check(_static(None), now - 5)                                      # нет детектора


def test_trim_answer_drops_unfinished_sentence():
    from aikoru.vlm import trim_answer

    assert trim_answer(" A bus. Two people are cro") == "A bus."
    assert trim_answer("A bus on the street") == "A bus on the street"
    assert trim_answer("Yes!") == "Yes!"


def test_resolve_vlm_backend(tmp_path):
    import os

    from aikoru.config import resolve_vlm_backend

    def cfg(**kw):
        return {"device": "cpu", "gguf": "g/m.gguf", "mmproj": "g/p.gguf", **kw}

    v = cfg()
    resolve_vlm_backend(v, tmp_path)
    assert v["backend"] == "transformers"  # файлов нет

    (tmp_path / "g").mkdir()
    (tmp_path / "llama.cpp").mkdir()
    for f in ("g/m.gguf", "g/p.gguf", "llama.cpp/" + ("llama-server.exe" if os.name == "nt" else "llama-server")):
        (tmp_path / f).write_bytes(b"")
    v = cfg()
    resolve_vlm_backend(v, tmp_path)
    assert v["backend"] == "llamacpp" and v["gguf"] == str(tmp_path / "g/m.gguf")
    v = cfg(device="cuda")
    resolve_vlm_backend(v, tmp_path)
    assert v["backend"] == "transformers"  # на GPU остаётся transformers
    v = cfg(backend="transformers")
    resolve_vlm_backend(v, tmp_path)
    assert v["backend"] == "transformers"


def test_calib_frames(tmp_path):
    from aikoru.detector import calib_frames

    assert calib_frames(None) == []
    assert calib_frames(tmp_path / "нет") == []
    for n in ("a.jpg", "b.PNG", "c.txt"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in calib_frames(tmp_path)] == ["a.jpg", "b.PNG"]
