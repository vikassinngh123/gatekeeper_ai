# 🛡️ GateKeeper AI — Edge-Cloud Hybrid Perimeter Security

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Vercel-black?style=for-the-badge&logo=vercel)](https://gatekeeper-ai-mu.vercel.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ByteTrack-FF6F00?style=for-the-badge)](https://github.com/vikassinngh123/gatekeeper_ai)

**GateKeeper AI** is a privacy-first, hybrid edge-cloud perimeter security prototype combining a multi-threaded local computer vision console (`edge/edge_engine.py` & `edge/local_edge_app.py`), a cloud incident API (`backend/main.py`), and a live Next.js command center (`frontend/`).

Heavy video inference and 128D biometric vector matching run strictly on-premise to keep raw CCTV feeds local, while verified breach events and lightweight evidence snapshots sync to the cloud dashboard in real time.

---

## 🏗️ System Architecture

```text
Local webcam / uploaded clip
          │
          ▼
  VideoSource capture thread (Buffer = 1)
          │
          ▼
 YOLOv8n person detector + ByteTrack (configurable stride & image size)
          │                         \
          ▼                          ▼
 Zone / dwell state + overlay      Asynchronous Face-ID worker queue (YuNet + SFace 128D)
          │                          │
          +------------+-------------+
                       │
             One annotated JPEG frame
                       │
        Localhost MJPEG viewer (127.0.0.1:8765) ──► Native browser <img> stream (20–35+ FPS)

Unverified person exceeds grace period ──► HTTP POST ──► Cloud FastAPI API ──► Database ──► Next.js Dashboard
Authorized person enters/leaves calibrated zone ──► edge/access_log.csv + Authorized Access Tab
```

---

## ✨ Key Engineering Highlights

* **⚡ Multi-Threaded 20–35+ FPS Edge Pipeline:** Decouples webcam capture (`VideoSource`), person tracking (`YOLOv8n + ByteTrack`), and biometric verification (`YuNet + SFace`) into asynchronous worker queues (`queue.Queue(maxsize=3)`) so facial recognition never blocks the video feed.
* **🎥 Zero-Lag Localhost MJPEG Streamer:** Bypasses Streamlit's slow `st.image()` WebSocket serialization by serving annotated frames over a local `ThreadingHTTPServer` (`multipart/x-mixed-replace` bound to `127.0.0.1`).
* **🔒 Privacy-First In-Memory Biometrics:** Converts reference enrollment photos into **128-dimensional SFace embeddings** and **SHA-256 digests** directly in RAM without saving uploaded biometric photos to disk.
* **📐 Dual Camera Geometry Modes:** Supports **Front-Facing Foot-Point Tracking** (`((bx1+bx2)//2, by2)`) and **Entrance-Wall Overhead Centroid Tracking** with configurable dwell-time grace periods and 3-second presence debouncing.
* **☁️ Decoupled Cloud Telemetry:** Compresses intruder crops under `170 KB` and dispatches non-blocking HTTP POST payloads to a FastAPI + SQLite/PostgreSQL backend polled by the [Next.js Command Center](https://gatekeeper-ai-mu.vercel.app).

---

## 📂 Repository Structure

```text
gatekeeper_ai/
├── backend/               # FastAPI cloud incident API & database models
│   ├── main.py
│   └── requirements.txt
├── edge/                  # Local Streamlit operator console & multi-threaded CV engine
│   ├── edge_engine.py
│   ├── local_edge_app.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/              # Next.js incident command dashboard (deployed on Vercel)
│   └── ...
├── render.yaml            # Render deployment blueprint for the FastAPI backend
└── README.md
```

---

## 🚀 Local Quick Start (Windows Terminal)

Open **three Windows Terminal tabs** at the repository root:

### Tab 1 — Cloud Incident API (`backend/`)
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
*Verify the API at `http://127.0.0.1:8000/` and interactive Swagger docs at `http://127.0.0.1:8000/docs`.*

### Tab 2 — Local Edge Vision Console (`edge/`)
```powershell
cd edge
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run local_edge_app.py
```
*Optionally copy `.env.example` to `.env` and configure `GATEKEEPER_API_URL` (or set it directly in the Streamlit sidebar). Open `http://localhost:8501`. First launch automatically downloads the required YOLO and OpenCV ONNX weights. Authorized entry/exit logs are written locally to `edge/access_log.csv`.*

### Tab 3 — Next.js Incident Dashboard (`frontend/`)
*(Requires Node.js LTS)*
```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```
*Open `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` in `.env.local` to point to your local or Render backend URL.*

---

## ⚙️ Performance & FPS Tuning Guide

1. **CPU Optimization:** Test with `Run YOLO every Nth frame = 2` (detection stride) and `Inference image size = 256` or `320`.
2. **Bottleneck Diagnostics:** Monitor **YOLO latency** and **Processing FPS** in the live telemetry cards.
3. **Stride vs. Resolution Trade-offs:** Compare `stride = 1, 2, 3` and `imgsz = 256, 320, 416`. Higher stride reuses bounding boxes between detection frames; smaller image sizes increase speed
