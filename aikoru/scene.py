"""Состояние сцены поверх треков ByteTrack: что появилось, что пропало.

ByteTrack иногда выдаёт новый номер тому же объекту (после перекрытия или
выхода из кадра). Чтобы не объявлять его заново, новый трек «наследует»
состояние недавно потерянного трека того же класса рядом с тем же местом.
"""
from __future__ import annotations

from dataclasses import dataclass

from .detector import Detection


@dataclass
class Track:
    id: int
    name: str
    det: Detection
    first_seen: float
    last_seen: float
    cx: float                       # центр по x, 0..1
    hits: int = 1
    announced: bool = False         # пользователю уже сообщили об объекте
    danger_level: str | None = None # последняя объявленная дистанция опасности
    identity: str | None = None     # имя знакомого человека
    face_tries: int = 0
    last_face_try: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


def _height(t: Track) -> float:
    return max(t.det.box[3] - t.det.box[1], 1.0)


class Scene:
    def __init__(self, lost_after_s: float = 1.5, reappear_window_s: float = 4.0):
        self.lost_after_s = lost_after_s
        self.reappear_window_s = reappear_window_s
        self.tracks: dict[int, Track] = {}
        self._recently_lost: list[tuple[float, Track]] = []

    def update(self, dets: list[Detection], now: float, frame_width: int) -> list[Track]:
        """Обновляет треки. Возвращает объекты, окончательно ушедшие из вида
        (потеряны и не вернулись за reappear_window_s)."""
        fresh = []
        for d in dets:
            cx = (d.box[0] + d.box[2]) / 2 / frame_width
            tr = self.tracks.get(d.track_id)
            if tr is None:
                fresh.append((d, cx))
            else:
                tr.det, tr.last_seen, tr.cx = d, now, cx
                tr.hits += 1
        for d, cx in fresh:  # новые номера — после обновления известных треков
            tr = Track(d.track_id, d.name, d, now, now, cx)
            self._inherit(tr, now)
            self.tracks[d.track_id] = tr

        lost = [t for t in self.tracks.values() if now - t.last_seen > self.lost_after_s]
        for t in lost:
            del self.tracks[t.id]
            if t.confirmed:
                self._recently_lost.append((now, t))
        gone = [t for ts, t in self._recently_lost if now - ts >= self.reappear_window_s]
        self._recently_lost = [(ts, t) for ts, t in self._recently_lost
                               if now - ts < self.reappear_window_s]
        return gone

    def visible(self, now: float) -> list[Track]:
        """Треки, видимые в текущем кадре."""
        return [t for t in self.tracks.values() if t.last_seen == now]

    def _inherit(self, tr: Track, now: float) -> None:
        """Ищет «предка» нового трека: тот же класс, рядом по x, и он либо недавно
        потерян, либо не обновился в этом кадре (ByteTrack сменил ему номер)."""
        candidates = [("lost", i, old) for i, (_, old) in enumerate(self._recently_lost)]
        candidates += [("stale", old.id, old) for old in self.tracks.values() if old.last_seen < now]
        candidates = [c for c in candidates if c[2].name == tr.name
                      and abs(c[2].cx - tr.cx) < (0.15 if c[0] == "stale" else 0.3)
                      and 0.6 < _height(tr) / _height(c[2]) < 1.6]
        if not candidates:
            return
        kind, key, old = min(candidates, key=lambda c: abs(c[2].cx - tr.cx))
        if kind == "lost":
            self._recently_lost.pop(key)
        else:
            del self.tracks[key]
        tr.announced, tr.identity = old.announced, old.identity
        tr.danger_level, tr.face_tries = old.danger_level, old.face_tries
        tr.last_face_try = old.last_face_try
        tr.hits = max(tr.hits, old.hits)
        tr.first_seen = old.first_seen
