"""Локальный веб-интерфейс AIKORU (только стандартная библиотека).

    GET  /                 интерфейс (aikoru/web/index.html)
    GET  /frame.jpg        текущий кадр камеры (интерфейс опрашивает его ~10 раз/с)
    GET  /stream.mjpg      видео с камеры (MJPEG, для VLC и т.п.)
    GET  /events           Server-Sent Events: произнесённые фразы, команды, треки
    GET  /api/state        состояние: язык, пауза, знакомые лица, треки
    POST /api/command      {"text": "...", "lang": "ru"|"kk"}
    POST /api/pause        {"paused": true|false}
    POST /api/faces/forget {"name": "..."}
"""
from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import lexicon
from .commands import parse

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "web"


class EventBus:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.history: list[dict] = []   # последние события для новых вкладок

    def publish(self, kind: str, **data) -> None:
        event = {"kind": kind, "ts": time.time(), **data}
        with self._lock:
            if kind != "tracks":
                self.history = (self.history + [event])[-50:]
            for q in self._subs:
                q.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subs.append(q)
            for e in self.history:
                q.put(e)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


class WebUI:
    def __init__(self, assistant, host: str = "127.0.0.1", port: int = 8765):
        self.a = assistant
        self.bus = EventBus()
        assistant.tts.listeners.append(
            lambda text, lang, prio: self.bus.publish("say", text=text, lang=lang, priority=prio)
        )
        handler = type("Handler", (_Handler,), {"ui": self})
        self.server = ThreadingHTTPServer((host, port), handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{port}"

    def start(self) -> None:
        threading.Thread(target=self.server.serve_forever, name="web", daemon=True).start()
        threading.Thread(target=self._tracks_loop, name="web-tracks", daemon=True).start()
        log.info("Веб-интерфейс: %s", self.url)

    def stop(self) -> None:
        self.server.shutdown()

    # --- данные --------------------------------------------------------------
    def tracks(self) -> list[dict]:
        lang = self.a.lang
        out = []
        for t in list(self.a.scene.tracks.values()):
            if not t.confirmed or time.time() - t.last_seen > 0.6:
                continue
            d = t.det
            out.append({
                "id": t.id, "name": t.name, "identity": t.identity, "box": d.box,
                "label": t.identity or lexicon.object_name(d.name, lang) or d.name,
                "where": f"{lexicon.PHRASES[lang][d.direction]}, "
                         f"{lexicon.distance_phrase(d.distance, d.meters, lang)}",
                "danger": d.name in self.a.cfg["detector"]["danger_classes"]
                          and d.direction == "center" and d.distance != "far",
            })
        return out

    def state(self) -> dict:
        return {
            "lang": self.a.lang,
            "paused": self.a.paused,
            "device": self.a.cfg["device"],
            "faces": sorted(self.a.faces.db) if self.a.faces else None,
            "tracks": self.tracks(),
        }

    def _tracks_loop(self) -> None:
        while True:
            self.bus.publish("tracks", tracks=self.tracks())
            time.sleep(0.25)


class _Handler(BaseHTTPRequestHandler):
    ui: WebUI

    def log_message(self, fmt, *args):  # без шума в консоли
        log.debug("web: " + fmt, *args)

    # --- утилиты -------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # --- GET -----------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path == "/stream.mjpg":
            return self._stream()
        if path == "/frame.jpg":
            ok, jpg = cv2.imencode(".jpg", self.ui.a.camera.read(), [cv2.IMWRITE_JPEG_QUALITY, 70])
            return self._send(200 if ok else 503, jpg.tobytes() if ok else b"", "image/jpeg")
        if path == "/events":
            return self._events()
        if path == "/api/state":
            return self._json(self.ui.state())
        f = (STATIC / path.lstrip("/")).resolve()
        if f.is_file() and STATIC.resolve() in f.parents:
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
                ctype += "; charset=utf-8"
            return self._send(200, f.read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                ok, jpg = cv2.imencode(".jpg", self.ui.a.camera.read(), [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                time.sleep(0.08)
        except (ConnectionError, OSError):
            pass

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = self.ui.bus.subscribe()
        try:
            while True:
                try:
                    e = q.get(timeout=15)
                    self.wfile.write(f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (ConnectionError, OSError):
            pass
        finally:
            self.ui.bus.unsubscribe(q)

    # --- POST ----------------------------------------------------------------
    def do_POST(self):
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        a = self.ui.a
        if self.path == "/api/command":
            text = str(body.get("text", "")).strip()
            lang = body.get("lang") if body.get("lang") in ("ru", "kk") else a.lang
            if not text:
                return self._json({"error": "empty"}, 400)
            cmd = parse(text, lang)
            a._commands.put(cmd)  # в ленту событий попадёт из Assistant.handle
            return self._json({"intent": cmd.intent})
        if self.path == "/api/pause":
            a.paused = bool(body.get("paused"))
            return self._json(self.ui.state())
        if self.path == "/api/faces/forget":
            if not a.faces:
                return self._json({"error": "faces disabled"}, 400)
            return self._json({"removed": a.faces.forget(str(body.get("name", "")) or None)})
        self._json({"error": "not found"}, 404)
