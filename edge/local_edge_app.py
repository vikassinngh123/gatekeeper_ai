from __future__ import annotations

import base64
import hashlib
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import streamlit_drawable_canvas as sdc
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

from edge_engine import Engine, PROC_H, PROC_W, start_mjpeg_server

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
TEMP_VIDEO_PATH = BASE_DIR / "temp_edge_input.mp4"
ACCESS_LOG_PATH = BASE_DIR / "access_log.csv"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = BASE_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

st.set_page_config(page_title="GateKeeper AI | Edge Console", page_icon="🛡️", layout="wide")

# Compatibility shim for streamlit-drawable-canvas on some Streamlit versions.
def _safe_canvas_image_to_url(image, *args, **kwargs):
    if isinstance(image, (bytes, bytearray)):
        b64 = base64.b64encode(image).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, Image.Image):
        from io import BytesIO
        buf = BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return ""

try:
    sdc.st_image.image_to_url = _safe_canvas_image_to_url
except Exception:
    pass

# Light is the default. The sidebar button switches this per browser session.
st.session_state.setdefault("gk_theme", "light")
THEMES = {
    "light": {
        "page": "#f4f7fb", "text": "#1f2d40", "muted": "#607086", "border": "#dce5ef",
        "card": "#ffffff", "panel": "#eef4fa", "hero_a": "#ffffff", "hero_b": "#e4f4f2",
        "hero_text": "#18283b", "accent": "#0f766e", "metric_value": "#1f2d40",
        "video_frame": "#edf2f7", "pipeline": "#f9fbfe",
    },
    "dark": {
        "page": "#0b1220", "text": "#e8eff8", "muted": "#96a7bb", "border": "#26364b",
        "card": "#101b2b", "panel": "#101b2b", "hero_a": "#101d31", "hero_b": "#102b2b",
        "hero_text": "#f3f7fc", "accent": "#69e4b4", "metric_value": "#f1f6ff",
        "video_frame": "#060b13", "pipeline": "#101b2b",
    },
}
palette = THEMES[st.session_state.gk_theme]
st.markdown(
    f"""
    <style>
      .stApp {{ background: {palette['page']}; color: {palette['text']};
                --background-color: {palette['page']}; --secondary-background-color: {palette['card']};
                --text-color: {palette['text']}; --primary-color: {palette['accent']}; }}
      [data-testid="stHeader"] {{ background: {palette['page']}; }}
      [data-testid="stSidebar"] {{ background: {palette['card']}; border-right: 1px solid {palette['border']}; }}
      .block-container {{ max-width: 1500px; padding-top: 1.3rem; padding-bottom: 2rem; }}
      .gk-hero {{ padding: 1.15rem 1.35rem; border: 1px solid {palette['border']}; border-radius: 18px;
                 background: linear-gradient(115deg, {palette['hero_a']} 0%, {palette['page']} 62%, {palette['hero_b']} 100%); }}
      .gk-eyebrow {{ color: {palette['accent']}; font-size: .76rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }}
      .gk-title {{ color: {palette['hero_text']}; font-size: 1.95rem; font-weight: 780; margin: .2rem 0 .25rem 0; }}
      .gk-subtitle {{ color: {palette['muted']}; font-size: .95rem; margin: 0; }}
      .gk-section {{ color: {palette['text']}; font-weight: 750; font-size: 1.04rem; margin: .45rem 0 .2rem 0; }}
      div[data-testid="stMetric"] {{ background: {palette['card']}; border: 1px solid {palette['border']}; padding: .72rem .85rem; border-radius: 14px; box-shadow: 0 2px 8px rgba(31,45,64,.035); }}
      div[data-testid="stMetricLabel"] {{ color: {palette['muted']}; }}
      div[data-testid="stMetricValue"] {{ color: {palette['metric_value']}; }}
      div[data-testid="stTabs"] button {{ font-weight: 650; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {palette['border']}; border-radius: 12px; overflow: hidden; }}
      .gk-muted {{ color: {palette['muted']}; font-size: .85rem; }}
      .gk-pipeline {{ color: {palette['text']}; background: {palette['pipeline']}; border: 1px solid {palette['border']}; border-radius: 12px; padding: .75rem .95rem; }}
      .gk-note {{ background: {palette['panel']}; color: {palette['text']}; border: 1px solid {palette['border']}; padding: .8rem 1rem; border-radius: 12px; }}
      .stButton > button, .stDownloadButton > button {{ border-radius: 10px; }}
      [data-testid="stFileUploader"] {{ background: {palette['card']}; border-radius: 12px; }}
      label, p, span {{ }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _download_model(path: Path, url: str) -> None:
    if not path.exists() or path.stat().st_size < 100_000:
        tmp = path.with_suffix(path.suffix + ".download")
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 100_000:
                raise RuntimeError(f"Downloaded model looks incomplete: {path.name}")
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


@st.cache_resource(show_spinner=False)
def load_runtime():
    """One local runtime shared by Streamlit reruns; intended for one local operator."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(os.environ.get("GATEKEEPER_YOLO_MODEL", "yolov8n.pt"))
    _download_model(YUNET_PATH, YUNET_URL)
    _download_model(SFACE_PATH, SFACE_URL)
    face_detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)
    face_recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    engine = Engine(yolo_model, face_detector, face_recognizer, device, log_path=str(ACCESS_LOG_PATH))
    stream_port = start_mjpeg_server(engine)
    return engine, device, stream_port


def read_video_frame(path: Path, frame_number: int = 0):
    if not path.exists():
        return None, 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(frame_number, 0), total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, total
    frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), total


def capture_webcam_frame(source=0):
    """Grab a one-off calibration frame from a local camera index or stream URL."""
    if isinstance(source, str) and source.strip().lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        cap = cv2.VideoCapture(source.strip())
    else:
        try:
            index = int(source)
        except (TypeError, ValueError):
            index = 0
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        frame = None
        # Give a network stream a moment to deliver a decoded frame; this is only
        # a calibration snapshot, not the continuous processing loop.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
                break
        if frame is None:
            return None
        frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def decode_uploaded_images(uploaded_files):
    images = []
    for item in uploaded_files or []:
        raw = np.frombuffer(item.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None and img.size:
            images.append(img)
    return images


@st.fragment(run_every=1.0)
def live_metrics(engine: Engine, device: str):
    live = dict(engine.live)
    stats = dict(engine.stats)
    cloud = dict(engine.cloud)
    c = st.columns(5)
    c[0].metric("Processing FPS", f"{stats['fps']:.1f}", help="Frames processed and published by the edge loop; not the source video's nominal FPS.")
    c[1].metric("YOLO latency", f"{stats['yolo_ms']:.0f} ms", help="Smoothed time spent in each detector/tracker call.")
    c[2].metric("People tracked", int(live["people"]))
    c[3].metric("Inside zone", int(live["in_zone"]))
    c[4].metric("Critical breaches", int(live["breaching"]))
    c2 = st.columns(4)
    c2[0].metric("Authorized inside", int(live["auth_in_zone"]))
    c2[1].metric("Face matching", f"{stats['face_ms']:.0f} ms", help="Smoothed face detector/embedding time when a face crop is processed.")
    c2[2].metric("Drawing + JPEG", f"{stats['draw_ms']:.0f} ms")
    cloud_label = "Connected" if cloud["state"] == "ok" else "Offline / unknown" if cloud["state"] in ("unknown", "error") else cloud["state"].title()
    c2[3].metric("Cloud API", cloud_label, help=cloud.get("detail", ""))
    if engine.error:
        st.error(engine.error)
    elif engine.running:
        st.success(f"Edge engine running · {device.upper()} · output stream is served separately from Streamlit.")
    else:
        st.info("Engine stopped. Configure the source and zone, then select Start monitoring.")


# Header first so the page gives immediate context during first model load.
st.markdown(
    """
    <div class="gk-hero">
      <div class="gk-eyebrow">Local edge node · privacy-aware prototype</div>
      <div class="gk-title">🛡️ GateKeeper AI</div>
      <p class="gk-subtitle">Live webcam and network camera monitoring, authorized-presence events, and cloud incident alerts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.spinner("Loading YOLO and face models on first run… this can take a few minutes."):
    try:
        engine, device, stream_port = load_runtime()
    except Exception as exc:
        st.error(f"The edge runtime could not start: {exc}")
        st.info("Check internet access for the first model download, the Python environment, and the package versions in edge/requirements.txt.")
        st.stop()

if "zone_rect" not in st.session_state:
    st.session_state.zone_rect = (140, 60, 500, 310)
if "reference_frame" not in st.session_state:
    st.session_state.reference_frame = None
if "video_sig" not in st.session_state:
    st.session_state.video_sig = None
if "video_total_frames" not in st.session_state:
    st.session_state.video_total_frames = 1
if "canvas_nonce" not in st.session_state:
    st.session_state.canvas_nonce = 0
if "video_path_ready" not in st.session_state:
    st.session_state.video_path_ready = False

# Focused configuration: only the settings that affect this local operator's run.
with st.sidebar:
    st.markdown("## ⚙️ Run configuration")
    theme_label = "☀️ Switch to light mode" if st.session_state.gk_theme == "dark" else "🌙 Switch to dark mode"
    if st.button(theme_label, key="gk_theme_toggle", width="stretch"):
        st.session_state.gk_theme = "light" if st.session_state.gk_theme == "dark" else "dark"
        st.rerun()
    st.caption(f"Edge device: **{device.upper()}**")
    source_choice = st.radio(
        "Video source",
        ["Local webcam", "Phone / IP camera", "Uploaded clip"],
        horizontal=False,
        help="Webcam and network streams are read by the local edge computer, not by Vercel.",
    )
    api_url = st.text_input("FastAPI endpoint", value=os.environ.get("GATEKEEPER_API_URL", "http://127.0.0.1:8000"))
    zone_name = st.text_input("Zone name", value="Server Room Alpha", max_chars=60)
    camera_mode = st.selectbox(
        "Camera angle",
        ["Front-Facing (Foot-Point Tracking)", "Entrance-Wall / Overhead (Centroid + Occlusion Grace)"],
    )
    grace_period = st.slider("Unverified-zone grace (seconds)", 1.0, 8.0, 3.5, 0.5)
    detect_stride = st.select_slider("Run YOLO every Nth frame", options=[1, 2, 3], value=2,
        help="Higher stride can improve throughput but uses older boxes between detector runs.")
    inference_size = st.select_slider("Inference image size", options=[256, 320, 416], value=320,
        help="256 is faster on many CPUs but may miss smaller/farther people; 416 can improve small-object detail at a speed cost.")
    blur_preview = st.checkbox("Blur top region of authorized person boxes", value=True,
        help="Best-effort visual privacy treatment only; it is not guaranteed face anonymization.")
    leave_debounce = st.slider("Exit confirmation delay (seconds)", 1.0, 8.0, 3.0, 0.5)
    loop_clip = st.checkbox("Loop uploaded clip", value=False)
    st.markdown("---")
    st.caption("For a fair FPS comparison, change one performance setting at a time and test the same short clip.")

# Clear stale calibration imagery when changing the type of source.
if st.session_state.get("last_source_choice") != source_choice:
    st.session_state.reference_frame = None
    st.session_state.last_source_choice = source_choice

# Source setup is kept above the tabs so users can configure a source before calibration.
source_path = ""
source_ready = False

if source_choice == "Uploaded clip":
    uploaded_video = st.file_uploader("Choose surveillance clip", type=["mp4", "avi", "mov", "mkv"], key="gk_video_upload")
    if uploaded_video is not None:
        video_bytes = uploaded_video.getvalue()
        sig = hashlib.sha256(video_bytes).hexdigest()
        if st.session_state.video_sig != sig or st.session_state.reference_frame is None:
            TEMP_VIDEO_PATH.write_bytes(video_bytes)
            st.session_state.video_sig = sig
            snap, total = read_video_frame(TEMP_VIDEO_PATH, 0)
            st.session_state.video_total_frames = total
            st.session_state.reference_frame = snap
            st.session_state.canvas_nonce += 1
        source_path = str(TEMP_VIDEO_PATH)
        source_ready = TEMP_VIDEO_PATH.exists() and st.session_state.reference_frame is not None
        if TEMP_VIDEO_PATH.exists():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                selected_frame = st.slider("Choose frame for zone calibration", 0, max(0, st.session_state.video_total_frames - 1), 0)
            with col_b:
                st.write("")
                if st.button("Load calibration frame", width="stretch"):
                    snap, _ = read_video_frame(TEMP_VIDEO_PATH, selected_frame)
                    if snap is not None:
                        st.session_state.reference_frame = snap
                        st.session_state.canvas_nonce += 1
elif source_choice == "Local webcam":
    source_device = st.selectbox("Camera device index", options=[0, 1, 2, 3], index=0,
                                 help="0 is usually the built-in/default webcam. Try 1 or 2 for USB cameras.")
    if st.session_state.get("gk_last_webcam_index") != source_device:
        st.session_state.reference_frame = None
        st.session_state.gk_last_webcam_index = source_device
    source_path = str(source_device)
    source_valid = True
    source_ready = st.session_state.reference_frame is not None
    cap_col, cap_btn = st.columns([3, 1])
    with cap_col:
        st.caption("Uses the webcam connected to the computer running this Streamlit app. Close other apps that may occupy it.")
    with cap_btn:
        if st.button("Capture calibration frame", width="stretch", key="gk_capture_webcam"):
            snap = capture_webcam_frame(source_device)
            if snap is None:
                st.error(f"Could not capture camera index {source_device}. Check camera permissions, try another index, and close other apps using the camera.")
            else:
                st.session_state.reference_frame = snap
                st.session_state.canvas_nonce += 1
else:
    source_path = st.text_input(
        "Phone / IP camera stream URL",
        value=os.environ.get("GATEKEEPER_CAMERA_URL", ""),
        placeholder="rtsp://camera-address:554/stream or http://phone-ip:port/video",
        help="Paste the stream URL supplied by your phone-camera app or IP camera. This URL is used only by the local edge computer.",
    ).strip()
    valid_scheme = source_path.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
    if st.session_state.get("gk_last_network_url") != source_path:
        st.session_state.reference_frame = None
        st.session_state.gk_last_network_url = source_path
    source_valid = bool(source_path and valid_scheme)
    source_ready = source_valid and st.session_state.reference_frame is not None
    st.markdown(
        '<div class="gk-note"><b>Connecting a phone camera</b><br>1. Install/open a phone camera-streaming app that exposes an RTSP or HTTP/MJPEG feed.<br>2. Put the phone and the computer running GateKeeper on the same Wi-Fi network.<br>3. Copy the actual stream URL from the app into the field above, then capture a calibration frame.<br>4. If the phone asks for camera/microphone access, grant only the permissions needed. Keep the stream on a trusted network.</div>',
        unsafe_allow_html=True,
    )
    if source_path and not valid_scheme:
        st.warning("Use a stream URL beginning with rtsp://, rtsps://, http://, or https://. A phone's ordinary web page URL is not necessarily its video stream.")
    net_col, net_btn = st.columns([3, 1])
    with net_col:
        st.caption("The local PC must be able to reach the phone/IP camera stream. Some RTSP feeds require an OpenCV build with FFmpeg support.")
    with net_btn:
        if st.button("Capture calibration frame", width="stretch", key="gk_capture_network", disabled=not source_valid):
            snap = capture_webcam_frame(source_path)
            if snap is None:
                st.error("Could not read a frame from that stream. Verify the exact RTSP/HTTP stream URL, the same-LAN connection, the camera app's streaming setting, and firewall rules.")
            else:
                st.session_state.reference_frame = snap
                st.session_state.canvas_nonce += 1

st.markdown("")
tab_live, tab_access, tab_people = st.tabs(["Live monitor", "Authorized access", "Personnel & model health"])

with tab_live:
    left, right = st.columns([1.05, 1.45], gap="large")
    with left:
        st.markdown('<div class="gk-section">1 · Draw the monitored zone</div>', unsafe_allow_html=True)
        st.caption("Drag a rectangle over the frame. The saved boundary is 640 × 360 processing coordinates.")
        if st.session_state.reference_frame is not None:
            bg = Image.fromarray(st.session_state.reference_frame)
        else:
            blank_rgb = (235, 242, 249) if st.session_state.gk_theme == "light" else (22, 30, 45)
            blank = np.full((PROC_H, PROC_W, 3), blank_rgb, dtype=np.uint8)
            label_colour = (95, 112, 134) if st.session_state.gk_theme == "light" else (160, 170, 185)
            cv2.putText(blank, "Select a source to calibrate", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, .65, label_colour, 2)
            bg = Image.fromarray(blank)

        canvas_result = st_canvas(
            fill_color="rgba(239, 68, 68, 0.22)", stroke_width=2, stroke_color="#ef4444",
            background_image=bg, update_streamlit=True, height=PROC_H, width=PROC_W,
            drawing_mode="rect", key=f"gk_zone_canvas_{st.session_state.canvas_nonce}",
        )
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            rects = [obj for obj in objects if isinstance(obj, dict) and obj.get("type") == "rect"]
            if rects:
                r = rects[-1]
                x1 = int(max(0, min(PROC_W, r.get("left", 0))))
                y1 = int(max(0, min(PROC_H, r.get("top", 0))))
                x2 = int(max(0, min(PROC_W, x1 + r.get("width", 0) * r.get("scaleX", 1))))
                y2 = int(max(0, min(PROC_H, y1 + r.get("height", 0) * r.get("scaleY", 1))))
                if x2 - x1 > 15 and y2 - y1 > 15:
                    st.session_state.zone_rect = (x1, y1, x2, y2)
        zx1, zy1, zx2, zy2 = st.session_state.zone_rect
        st.caption(f"Current boundary: ({zx1}, {zy1}) → ({zx2}, {zy2})")
        if st.button("Reset boundary to default", width="stretch"):
            st.session_state.zone_rect = (140, 60, 500, 310)
            st.session_state.canvas_nonce += 1
            st.rerun()

    with right:
        st.markdown('<div class="gk-section">2 · Live video and controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="gk-pipeline">Capture → YOLO + ByteTrack → zone & dwell rules ··· Face-ID worker → access log / cloud alert</div>', unsafe_allow_html=True)
        st.write("")
        start_col, stop_col = st.columns(2)
        with start_col:
            start_clicked = st.button("▶ Start monitoring", type="primary", disabled=(not source_ready or engine.running), width="stretch")
        with stop_col:
            stop_clicked = st.button("■ Stop monitoring", disabled=(not engine.running), width="stretch")
        if not source_ready:
            st.info("Choose a source and capture/load a calibration frame before starting. Draw the restricted zone on that frame.")

        if start_clicked:
            engine.cfg.update({
                "api_url": api_url.strip().rstrip("/"), "grace": float(grace_period),
                "stride": int(detect_stride), "imgsz": int(inference_size),
                "blur": bool(blur_preview), "leave_debounce": float(leave_debounce),
            })
            engine.start(
                kind=("file" if source_choice == "Uploaded clip" else "network" if source_choice == "Phone / IP camera" else "webcam"),
                path=source_path, zone=tuple(st.session_state.zone_rect), zone_name=zone_name.strip() or "Monitored Zone",
                camera_mode=camera_mode, loop=bool(loop_clip),
            )
            st.rerun()
        if stop_clicked:
            engine.stop()
            st.rerun()

        live_metrics(engine, device)
        components.html(
            f"""<div style='background:{palette['video_frame']};border:1px solid {palette['border']};border-radius:14px;padding:8px;'>
            <img src='http://127.0.0.1:{stream_port}/stream.mjpg' style='display:block;width:100%;height:auto;min-height:220px;object-fit:contain;border-radius:8px;' />
            </div>""",
            height=390,
            scrolling=False,
        )
        st.caption("The video viewer refreshes independently of Streamlit. If the stream does not appear, allow localhost traffic in the browser and check whether port 8765 was blocked; the engine chooses another free port if needed.")

with tab_access:
    st.markdown('<div class="gk-section">Authorized zone-presence history</div>', unsafe_allow_html=True)
    st.caption("This table contains names and event metadata only—no snapshots or image columns. ENTERED/LEFT refer to the configured rectangular zone, not a verified physical door crossing.")
    inside = engine.inside_now()
    k1, k2 = st.columns(2)
    k1.metric("Currently recorded inside", len(inside))
    k2.metric("Logged events", len(engine.get_access_log()))
    if inside:
        st.write("**Currently inside:** " + ", ".join(inside))
    rows = engine.get_access_log()
    if rows:
        df = pd.DataFrame(rows)
        column_order = [c for c in ["time", "person", "event", "zone", "duration_sec", "track_id"] if c in df.columns]
        df = df[column_order].rename(columns={
            "time": "Time", "person": "Name", "event": "Event", "zone": "Zone",
            "duration_sec": "Duration (s)", "track_id": "Track ID",
        })
        df = df.iloc[::-1].reset_index(drop=True)
        filter_col, refresh_col = st.columns([3, 1])
        with filter_col:
            event_filter = st.multiselect("Event type", ["ENTERED", "LEFT"], default=["ENTERED", "LEFT"])
        with refresh_col:
            st.write("")
            st.button("Refresh table", width="stretch", key="refresh_access_table")
        filtered = df[df["Event"].isin(event_filter)]
        st.dataframe(filtered, width="stretch", hide_index=True)
        st.download_button("Download access log CSV", data=engine.log_csv_bytes(), file_name="gatekeeper_authorized_access.csv", mime="text/csv", width="stretch")
        with st.expander("Clear local access history"):
            st.warning("This deletes access_log.csv from this machine. It does not delete cloud incidents.")
            if st.button("Clear access log", key="clear_access_log", type="secondary"):
                engine.clear_log()
                st.rerun()
    else:
        st.info("No authorized-zone entry/exit events yet. Enroll a test person, start monitoring, and let that person enter the calibrated zone.")

with tab_people:
    st.markdown('<div class="gk-section">Enroll a person</div>', unsafe_allow_html=True)
    st.caption("Use only images you have permission to process. Images are decoded in memory to create face embeddings; the app does not write the uploaded reference images to disk.")
    pcol, ucol = st.columns([1, 1.3], gap="large")
    with pcol:
        person_name = st.text_input("Name or staff ID", max_chars=80)
        person_images = st.file_uploader("Reference face photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="gk_enroll_photos")
        if st.button("Enroll face samples", type="primary", disabled=not (person_name.strip() and person_images), width="stretch"):
            imgs = decode_uploaded_images(person_images)
            if not imgs:
                st.error("No valid images could be decoded.")
            else:
                accepted, total = engine.enroll(person_name.strip(), imgs)
                if accepted:
                    st.success(f"Created {accepted} usable face embedding(s) from {total} image(s).")
                else:
                    st.warning("No usable face was detected. Try a well-lit, frontal, consented photo where the face is large and unobstructed.")
    with ucol:
        st.markdown("**Registered people**")
        roster = engine.roster()
        if roster:
            roster_df = pd.DataFrame(roster)[["name", "samples", "added"]].rename(columns={"name": "Name", "samples": "Samples", "added": "Added"})
            st.dataframe(roster_df, width="stretch", hide_index=True)
            remove_name = st.selectbox("Remove a person", options=[r["name"] for r in roster], key="gk_remove_name")
            if st.button("Remove selected person", width="stretch"):
                engine.remove_person(remove_name)
                st.rerun()
        else:
            st.info("No personnel enrolled in this session yet.")
        st.caption("The in-memory roster is lost when the Streamlit process restarts. The access log is saved separately to edge/access_log.csv.")

    st.markdown("---")
    st.markdown('<div class="gk-section">Performance and model notes</div>', unsafe_allow_html=True)
    st.write(f"**Runtime:** {device.upper()} · **Processing canvas:** {PROC_W} × {PROC_H} · **Inference size:** configured to {inference_size} · **Detection stride:** {detect_stride}")
    st.write("**How to improve FPS:** start with stride 2 and image size 256 on CPU; compare YOLO latency on the same clip. If YOLO latency stays high, use a CUDA-enabled PyTorch build on a supported NVIDIA GPU. MJPEG avoids sending every frame through Streamlit's image serializer, but it cannot make a slow detector itself run faster.")
    st.warning("Face matching is a prototype signal, not guaranteed identity proof. The threshold needs testing on your own consented dataset. The preview blur is best-effort and is not a promise of anonymization.")
