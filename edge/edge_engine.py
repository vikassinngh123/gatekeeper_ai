"""GateKeeper AI edge engine.

Everything that touches video lives here, in plain threads, with NO Streamlit
imports. The Streamlit app only starts/stops the engine and reads its state.

Pipeline (each stage runs independently, so a slow one never stalls the others):

    capture  ->  YOLO + ByteTrack  ->  draw + JPEG  ->  browser (MJPEG or st.image)
                       |
                       +--> face-ID worker thread (YuNet + SFace, throttled per track)
                       +--> cloud alert thread (HTTP POST)
"""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import requests

PROC_W, PROC_H = 640, 360          # processing resolution (zone coordinates live here)
LOG_PATH = "access_log.csv"        # names + timestamps only, never images
LOG_FIELDS = ["time", "person", "event", "zone", "duration_sec", "track_id"]
MAX_SNAPSHOT_BYTES = 170_000       # backend rejects > 180_000 decoded bytes

# BGR colours
GREEN, AMBER, ORANGE, RED, WHITE, PANEL = (
    (80, 200, 80), (0, 200, 255), (230, 160, 0), (60, 60, 240), (255, 255, 255), (42, 23, 15),
)


# --------------------------------------------------------------------------
# Video source
# --------------------------------------------------------------------------
class VideoSource:
    """Webcam (latest-frame reader thread) or video file (paced to real time).

    read() returns a frame, or None. For files, `ended` becomes True at EOF.
    Files are played in real time and frames are DROPPED when processing is
    slower than the clip, so breach timers (wall clock) match what you see.
    """

    def __init__(self, kind: str, path: str | None = None, loop: bool = False):
        """Open a clip, local camera index, or IP/phone camera URL.

        kind='file' uses a file path; kind='webcam' uses path as the numeric
        device index (default 0); kind='network' opens path as an HTTP(S)/RTSP
        camera stream. Network stream support depends on the installed OpenCV
        video backends and the URL provided by the camera/phone app.
        """
        self.is_file = kind == "file"
        self.loop = loop
        self.ended = False
        self.looped = False
        self._alive = True

        if self.is_file:
            self.cap = cv2.VideoCapture(path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if 1 < fps < 240 else 25.0
            self._t0 = time.time()
            self._pos = 0
        else:
            if kind == "network":
                source = str(path or "").strip()
                self.cap = cv2.VideoCapture(source)
            else:
                try:
                    source = int(path) if path not in (None, "") else 0
                except (TypeError, ValueError):
                    source = 0
                if os.name == "nt":
                    self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROC_W)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_H)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            # A small capture buffer prioritizes the newest frame for low latency.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._frame, self._seq, self._last_seq = None, 0, 0
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()
        self.opened = bool(self.cap.isOpened())

    # webcam ---------------------------------------------------------------
    def _grab_loop(self):
        while self._alive:
            ok, frame = self.cap.read()
            if ok:
                self._frame, self._seq = frame, self._seq + 1
            else:
                time.sleep(0.01)

    # shared ---------------------------------------------------------------
    def read(self):
        return self._read_file() if self.is_file else self._read_cam()

    def _read_cam(self):
        t0 = time.time()
        while self._seq == self._last_seq and self._alive and time.time() - t0 < 1.0:
            time.sleep(0.002)
        if self._seq == self._last_seq:
            return None
        self._last_seq = self._seq
        return self._frame

    def _read_file(self):
        target = int((time.time() - self._t0) * self.fps)
        while self._pos < target:                      # drop frames to stay real-time
            if not self.cap.grab():
                return self._rewind_or_end()
            self._pos += 1
        ok, frame = self.cap.read()
        if not ok:
            return self._rewind_or_end()
        self._pos += 1
        wait = self._pos / self.fps - (time.time() - self._t0)
        if wait > 0:
            time.sleep(min(wait, 0.1))
        return frame

    def _rewind_or_end(self):
        if self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._t0, self._pos, self.looped = time.time(), 0, True
            ok, frame = self.cap.read()
            if ok:
                self._pos = 1
                return frame
        self.ended = True
        return None

    def release(self):
        self._alive = False
        if not self.is_file and getattr(self, "_thread", None):
            self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, yolo, detector, recognizer, device: str, log_path: str = LOG_PATH):
        self.yolo, self.detector, self.recognizer, self.device = yolo, detector, recognizer, device
        self.log_path = log_path
        self.lock = threading.RLock()        # guards shared dicts/lists
        self.face_lock = threading.Lock()    # YuNet/SFace are not thread-safe

        # live-tunable settings (the UI updates these on every rerun)
        self.cfg = {
            "grace": 3.5, "stride": 2, "imgsz": 320, "conf": 0.35,
            "blur": True, "match_thr": 0.363, "leave_debounce": 3.0,
            "api_url": "http://127.0.0.1:8000",
        }

        self.enrolled: dict[str, dict] = {}          # name -> {vectors, sha256, added}
        self.access_log: list[dict] = self._load_log()

        self.running = False
        self.error: str | None = None
        self.cloud = {"state": "unknown", "detail": ""}
        self.stats = {"fps": 0.0, "yolo_ms": 0.0, "face_ms": 0.0, "draw_ms": 0.0, "frames": 0}
        self._stop = threading.Event()
        self._thread = self._face_thread = None
        self.face_q: queue.Queue = queue.Queue(maxsize=3)

        # frame hand-off to the viewers
        self.frame_cond = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.frame_seq = 0
        self.idle_jpeg = self._make_idle_jpeg()

        self._reset_runtime()

    # ---- state helpers ---------------------------------------------------
    def _reset_runtime(self):
        with self.lock:
            self.verified: dict[int, str] = {}
            self.breach_t0: dict[int, float] = {}
            self.alerted: set[int] = set()
            self.last_seen: dict[int, float] = {}
            self.face_last_try: dict[int, float] = {}
            self.inside_since: dict[str, float] = {}
            self.inside_last: dict[str, float] = {}
            self.inside_tid: dict[str, int] = {}
            self.zone_out_since: dict[int, float] = {}
            self.alerts: list[dict] = []
            self.live = {"people": 0, "in_zone": 0, "auth_in_zone": 0, "breaching": 0}

    @staticmethod
    def _make_idle_jpeg() -> bytes:
        img = np.zeros((PROC_H, PROC_W, 3), np.uint8) + 22
        cv2.putText(img, "ENGINE IDLE", (215, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        cv2.putText(img, "Press Start to begin surveillance", (170, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def _ema(self, key: str, value: float, alpha: float = 0.15):
        old = self.stats[key]
        self.stats[key] = value if old == 0 else (1 - alpha) * old + alpha * value

    # ---- face embedding (shared by enrolment and live verification) ------
    def embed(self, img_bgr, max_side: int = 320):
        """Return a 128-d SFace embedding of the best face in the image, else None."""
        if img_bgr is None or img_bgr.size == 0:
            return None
        h, w = img_bgr.shape[:2]
        if min(h, w) < 24:
            return None
        longest = max(h, w)
        scale = max_side / longest if longest > max_side else (160 / longest if longest < 160 else 1.0)
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)
        else:
            img = np.ascontiguousarray(img_bgr)
        ih, iw = img.shape[:2]
        t = time.perf_counter()
        with self.face_lock:
            self.detector.setInputSize((iw, ih))
            _, faces = self.detector.detect(img)
            if faces is None or len(faces) == 0:
                self._ema("face_ms", (time.perf_counter() - t) * 1000)
                return None
            face = faces[int(np.argmax(faces[:, -1]))]      # highest-confidence face
            feat = self.recognizer.feature(self.recognizer.alignCrop(img, face))
        self._ema("face_ms", (time.perf_counter() - t) * 1000)
        return feat

    def _best_match(self, vec):
        with self.lock:
            people = [(n, list(r["vectors"])) for n, r in self.enrolled.items()]
        best_name, best = None, -1.0
        for name, vectors in people:
            for ref in vectors:
                s = float(self.recognizer.match(vec, ref, cv2.FaceRecognizerSF_FR_COSINE))
                if s > best:
                    best_name, best = name, s
        return (best_name, best) if best >= self.cfg["match_thr"] else (None, best)

    # ---- enrolment -------------------------------------------------------
    def enroll(self, name: str, images_bgr: list) -> tuple[int, int]:
        vecs = [v for v in (self.embed(im, max_side=640) for im in images_bgr) if v is not None]
        if vecs:
            with self.lock:
                rec = self.enrolled.setdefault(name, {"vectors": [], "added": datetime.now()})
                rec["vectors"].extend(vecs)
                rec["sha256"] = hashlib.sha256(b"".join(v.tobytes() for v in rec["vectors"])).hexdigest()
        return len(vecs), len(images_bgr)

    def remove_person(self, name: str):
        with self.lock:
            self.enrolled.pop(name, None)
            self.verified = {t: n for t, n in self.verified.items() if n != name}

    def roster(self) -> list[dict]:
        with self.lock:
            return [
                {"name": n, "samples": len(r["vectors"]),
                 "added": r["added"].strftime("%Y-%m-%d %H:%M"), "digest": r["sha256"][:12]}
                for n, r in self.enrolled.items()
            ]

    # ---- access log (names + timestamps only) -----------------------------
    def _load_log(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                r["duration_sec"] = float(r["duration_sec"]) if r.get("duration_sec") else None
            return rows
        except Exception:
            return []

    def _log_event(self, ts: float, person: str, event: str, zone: str, duration, track_id):
        row = {
            "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "person": person, "event": event, "zone": zone,
            "duration_sec": duration, "track_id": track_id,
        }
        with self.lock:
            self.access_log.append(row)
            new_file = not os.path.exists(self.log_path)
            try:
                with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                    if new_file:
                        w.writeheader()
                    w.writerow({k: ("" if v is None else v) for k, v in row.items()})
            except OSError:
                pass

    def get_access_log(self) -> list[dict]:
        with self.lock:
            return list(self.access_log)

    def inside_now(self) -> list[str]:
        with self.lock:
            return sorted(self.inside_since)

    def clear_log(self):
        with self.lock:
            self.access_log.clear()
            if os.path.exists(self.log_path):
                os.remove(self.log_path)

    def log_csv_bytes(self) -> bytes:
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=LOG_FIELDS)
        w.writeheader()
        for r in self.get_access_log():
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in LOG_FIELDS})
        return buf.getvalue().encode("utf-8")

    def _update_presence(self, inside: dict[str, int], now: float, zone: str):
        """inside: {name: track_id} of authorized people in the zone this frame.

        Presence is keyed by NAME, not track id, so a ByteTrack ID switch does
        not create a fake LEFT + ENTERED pair. A person only 'leaves' after
        being absent for `leave_debounce` seconds.
        """
        debounce = self.cfg["leave_debounce"]
        with self.lock:
            for name, tid in inside.items():
                self.inside_last[name] = now
                if name not in self.inside_since:
                    self.inside_since[name], self.inside_tid[name] = now, tid
                    self._log_event(now, name, "ENTERED", zone, None, tid)
            for name in list(self.inside_since):
                if name not in inside and now - self.inside_last[name] >= debounce:
                    t_in, t_last = self.inside_since.pop(name), self.inside_last.pop(name)
                    self._log_event(t_last, name, "LEFT", zone, round(t_last - t_in, 1),
                                    self.inside_tid.pop(name, None))

    def _close_presence(self, zone: str):
        with self.lock:
            for name in list(self.inside_since):
                t_in, t_last = self.inside_since.pop(name), self.inside_last.pop(name, time.time())
                self._log_event(t_last, name, "LEFT", zone, round(t_last - t_in, 1), self.inside_tid.pop(name, None))

    # ---- lifecycle -------------------------------------------------------
    def start(self, kind: str, path: str, zone: tuple, zone_name: str, camera_mode: str, loop: bool = False):
        if self.running:
            return
        src = VideoSource(kind, path, loop)
        if not src.opened:
            self.error = "Could not open the video source (webcam busy, or unreadable file)."
            src.release()
            return
        self.error = None
        self._reset_runtime()
        self.stats.update(fps=0.0, yolo_ms=0.0, face_ms=0.0, draw_ms=0.0, frames=0)
        while not self.face_q.empty():
            try:
                self.face_q.get_nowait()
            except queue.Empty:
                break
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, args=(src, zone, zone_name, camera_mode), daemon=True)
        self._face_thread = threading.Thread(target=self._face_loop, daemon=True)
        self._thread.start()
        self._face_thread.start()

    def stop(self):
        self._stop.set()
        for t in (self._thread, self._face_thread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=3.0)

    # ---- face-ID worker --------------------------------------------------
    def _face_loop(self):
        while not self._stop.is_set():
            try:
                tid, crop = self.face_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # Drop queued crops for tracks that have disappeared. This avoids
                # assigning a delayed face result to a recycled tracker ID.
                with self.lock:
                    already_verified = tid in self.verified
                    last_seen = self.last_seen.get(tid, 0.0)
                if already_verified or time.time() - last_seen > 8.0:
                    continue
                vec = self.embed(crop)
                if vec is None:
                    continue
                name, _score = self._best_match(vec)
                if name:
                    with self.lock:
                        if time.time() - self.last_seen.get(tid, 0.0) <= 8.0:
                            self.verified[tid] = name
            except Exception:
                traceback.print_exc()

    @staticmethod
    def _head_crop(raw, x1, y1, x2, y2, sx, sy):
        """Upper 45% of the person box, cut from the FULL-RES frame (better faces)."""
        bw, bh = x2 - x1, y2 - y1
        rx1, rx2 = (x1 - 0.1 * bw) * sx, (x2 + 0.1 * bw) * sx
        ry1, ry2 = y1 * sy, (y1 + 0.45 * bh) * sy
        H, W = raw.shape[:2]
        rx1 = max(0, min(W, int(rx1)))
        rx2 = max(0, min(W, int(rx2)))
        ry1 = max(0, min(H, int(ry1)))
        ry2 = max(0, min(H, int(ry2)))
        crop = raw[ry1:ry2, rx1:rx2]
        return crop.copy() if crop.size else None

    # ---- cloud alert -----------------------------------------------------
    def _dispatch_alert(self, tid: int, dwell: float, crop, zone_name: str, camera_mode: str):
        quality, img = 80, crop
        buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        while len(buf) > MAX_SNAPSHOT_BYTES and min(img.shape[:2]) > 40:
            img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
            buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        payload = {
            "track_id": int(tid), "zone_name": zone_name, "threat_level": "CRITICAL BREACH",
            "camera_mode": camera_mode.split(" ")[0], "dwell_time_sec": float(dwell),
            "snapshot_base64": base64.b64encode(buf.tobytes()).decode("utf-8"),
        }
        alert = {"time": datetime.now().strftime("%H:%M:%S"), "track_id": int(tid),
                 "zone": zone_name, "dwell_sec": float(dwell), "cloud": "sending"}
        with self.lock:
            self.alerts.insert(0, alert)
            del self.alerts[50:]
        url = self.cfg["api_url"].rstrip("/") + "/api/incidents"

        def _post():
            try:
                r = requests.post(url, json=payload, timeout=3)
                ok = r.status_code == 201
                alert["cloud"] = "sent" if ok else f"failed ({r.status_code})"
                self.cloud = {"state": "ok" if ok else "error", "detail": f"HTTP {r.status_code}"}
            except Exception as exc:
                alert["cloud"] = "offline"
                self.cloud = {"state": "error", "detail": str(exc)[:80]}

        threading.Thread(target=_post, daemon=True).start()

    # ---- main pipeline ---------------------------------------------------
    def _run(self, src: VideoSource, zone: tuple, zone_name: str, camera_mode: str):
        zx1, zy1, zx2, zy2 = zone
        center_mode = "Entrance-Wall" in camera_mode
        use_half = self.device != "cpu"
        detections: list[tuple] = []
        frame_idx, prev = 0, time.perf_counter()
        try:
            if getattr(self.yolo, "predictor", None) is not None:
                self.yolo.predictor = None               # fresh ByteTrack state
            while not self._stop.is_set():
                raw = src.read()
                if raw is None:
                    if src.ended:
                        break
                    continue
                if src.looped:
                    src.looped = False
                    detections = []
                    if getattr(self.yolo, "predictor", None) is not None:
                        self.yolo.predictor = None
                frame_idx += 1
                now = time.time()
                frame = cv2.resize(raw, (PROC_W, PROC_H))
                clean = frame.copy()                      # un-annotated copy for evidence crops
                sx, sy = raw.shape[1] / PROC_W, raw.shape[0] / PROC_H

                # 1) detect + track (every `stride` frames; boxes are cached in between)
                stride = max(1, int(self.cfg["stride"]))
                if stride == 1 or (frame_idx - 1) % stride == 0 or not detections:
                    t = time.perf_counter()
                    imgsz = int(self.cfg.get("imgsz", 320))
                    conf = float(self.cfg.get("conf", 0.35))
                    res = self.yolo.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml",
                                          imgsz=imgsz, conf=conf, max_det=20, device=self.device,
                                          half=use_half, verbose=False)
                    self._ema("yolo_ms", (time.perf_counter() - t) * 1000)
                    detections = []
                    if res and res[0].boxes is not None and res[0].boxes.id is not None:
                        boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
                        ids = res[0].boxes.id.cpu().numpy().astype(int)
                        for b, tid in zip(boxes, ids):
                            detections.append((max(0, b[0]), max(0, b[1]), min(PROC_W, b[2]), min(PROC_H, b[3]), int(tid)))

                # 2) zone overlay
                t_draw = time.perf_counter()
                ov = frame.copy()
                cv2.rectangle(ov, (zx1, zy1), (zx2, zy2), (0, 0, 220), -1)
                cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
                cv2.putText(frame, f"ZONE: {zone_name.upper()}", (zx1 + 8, max(20, zy1 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                draw_ms = (time.perf_counter() - t_draw) * 1000

                # 3) per-person logic
                inside_auth: dict[str, int] = {}
                in_zone_n = breaching = 0
                for (bx1, by1, bx2, by2, tid) in detections:
                    px, py = ((bx1 + bx2) // 2, (by1 + by2) // 2) if center_mode else ((bx1 + bx2) // 2, by2)
                    in_zone = zx1 <= px <= zx2 and zy1 <= py <= zy2
                    self.last_seen[tid] = now
                    # Keep dwell/alert state through brief box jitter at the zone edge.
                    if in_zone:
                        self.zone_out_since.pop(tid, None)
                    else:
                        self.zone_out_since.setdefault(tid, now)
                        if now - self.zone_out_since[tid] >= float(self.cfg["leave_debounce"]):
                            self.breach_t0.pop(tid, None)
                            self.alerted.discard(tid)

                    with self.lock:
                        name = self.verified.get(tid)
                    if (name is None and by2 - by1 >= 40
                            and now - self.face_last_try.get(tid, 0.0) > 0.7):
                        crop = self._head_crop(raw, bx1, by1, bx2, by2, sx, sy)
                        if crop is not None:
                            try:
                                self.face_q.put_nowait((tid, crop))
                                self.face_last_try[tid] = now
                            except queue.Full:
                                pass

                    if name:
                        color = GREEN
                        label = f"#{tid} {name} [AUTHORIZED]"
                        if in_zone:
                            inside_auth[name] = tid
                        if self.cfg["blur"]:
                            hh = max(10, (by2 - by1) // 3)
                            region = frame[by1:by1 + hh, bx1:bx2]
                            if region.size:
                                frame[by1:by1 + hh, bx1:bx2] = cv2.GaussianBlur(region, (35, 35), 20)
                    elif not in_zone:
                        color, label = ORANGE, f"#{tid} outside zone"
                    else:
                        in_zone_n += 1
                        t0 = self.breach_t0.setdefault(tid, now)
                        dwell = round(now - t0, 1)
                        if dwell < self.cfg["grace"]:
                            color, label = AMBER, f"#{tid} unverified ({dwell}s)"
                        else:
                            color, label = RED, f"#{tid} CRITICAL BREACH ({dwell}s)"
                            breaching += 1
                            if tid not in self.alerted:
                                crop = clean[by1:by2, bx1:bx2]
                                if crop.size:
                                    self.alerted.add(tid)
                                    self._dispatch_alert(tid, dwell, crop, zone_name, camera_mode)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cv2.circle(frame, (px, py), 4, color, -1)
                    cv2.putText(frame, label, (bx1, max(18, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2)

                self._update_presence(inside_auth, now, zone_name)
                self.live = {"people": len(detections), "in_zone": in_zone_n + len(inside_auth),
                             "auth_in_zone": len(inside_auth), "breaching": breaching}

                # 4) HUD + publish one JPEG that every viewer reuses
                t_enc = time.perf_counter()
                cur = time.perf_counter()
                self._ema("fps", 1.0 / max(1e-4, cur - prev))
                prev = cur
                cv2.rectangle(frame, (8, 8), (205, 32), PANEL, -1)
                cv2.putText(frame, f"{self.stats['fps']:.1f} FPS | YOLO {self.stats['yolo_ms']:.0f}ms",
                            (13, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 230, 110), 1)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self._ema("draw_ms", draw_ms + (time.perf_counter() - t_enc) * 1000)
                self.stats["frames"] += 1
                if ok:
                    with self.frame_cond:
                        self.latest_jpeg = buf.tobytes()
                        self.frame_seq += 1
                        self.frame_cond.notify_all()

                if frame_idx % 90 == 0:                     # prune state of long-gone tracks
                    with self.lock:
                        expired = [t for t, seen in self.last_seen.items() if now - seen > 15]
                        for tid in expired:
                            for d in (self.last_seen, self.breach_t0, self.face_last_try, self.verified, self.zone_out_since):
                                d.pop(tid, None)
                            self.alerted.discard(tid)
        except Exception as exc:
            traceback.print_exc()
            self.error = f"Engine crashed: {exc!r}"
        finally:
            src.release()
            self._close_presence(zone_name)
            with self.frame_cond:
                self.latest_jpeg = None
                self.frame_seq += 1
                self.frame_cond.notify_all()
            self.live = {"people": 0, "in_zone": 0, "auth_in_zone": 0, "breaching": 0}
            self._stop.set()                                  # also stops the face worker
            self.running = False


# --------------------------------------------------------------------------
# Optional MJPEG viewer: bypasses Streamlit's per-frame websocket/media-file
# overhead, so the browser gets the stream at the engine's real frame rate.
# Bound to localhost only (it is a surveillance feed).
# --------------------------------------------------------------------------
def start_mjpeg_server(engine: Engine, host: str = "127.0.0.1", port: int = 8765) -> int:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, private")
            self.end_headers()
            last = -1
            try:
                while True:
                    with engine.frame_cond:
                        engine.frame_cond.wait_for(lambda: engine.frame_seq != last, timeout=1.0)
                        jpg, last = engine.latest_jpeg, engine.frame_seq
                    jpg = jpg or engine.idle_jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except OSError:          # browser closed the tab / navigated away
                return

        def log_message(self, *args):
            pass

    ThreadingHTTPServer.daemon_threads = True
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        server = ThreadingHTTPServer((host, 0), Handler)     # port taken: pick a free one
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]
