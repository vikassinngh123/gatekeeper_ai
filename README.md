# GateKeeper AI — hackathon prototype

GateKeeper AI is a local edge vision demo with a lightweight cloud incident API and a separate web dashboard. This ZIP includes a cleaner Streamlit operator console built around `edge/edge_engine.py`.

## Architecture

```text
Local webcam / uploaded clip
          |
          v
  VideoSource capture thread
          |
          v
 YOLO person detector + ByteTrack (configured stride and image size)
          |                         \
          v                          v
 zone/dwell state + overlay      face-ID worker queue
          |                          |
          +------------+-------------+
                       |
             one annotated JPEG frame
                       |
        localhost MJPEG viewer (port selected at startup)

Unverified person exceeds grace period -> HTTP POST -> backend incident API -> database -> Next.js dashboard
Authorized person enters/leaves calibrated rectangle -> access_log.csv + Authorized access tab
```

The edge app processes video on the computer running Streamlit; the cloud backend stores breach events and the Next.js dashboard polls the incident API. The MJPEG stream is intentionally bound to `127.0.0.1` and is not a public CCTV endpoint.

## Local start (Windows Terminal)

Use three Windows Terminal tabs opened at the repository root.

### Tab 1 — API backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/` and `http://127.0.0.1:8000/docs`.

### Tab 2 — local edge app

```powershell
cd edge
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run local_edge_app.py
```

Optionally copy `.env.example` to `.env` and edit `GATEKEEPER_API_URL`; the sidebar can also set it. Open the local Streamlit URL shown in the terminal, normally `http://localhost:8501`. The first launch downloads YOLO/OpenCV models and can take a few minutes. The enrolled roster is in memory and must be re-enrolled after a process restart. Authorized entry/exit logs are written to `edge/access_log.csv`.

### Tab 3 — Next.js incident dashboard

Install Node.js LTS first, then:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. If `.env.local` already exists, edit it instead of overwriting it. Set `NEXT_PUBLIC_API_URL` to the API URL and restart/redeploy the frontend after changing it.

## FPS tuning

1. Try the same short clip and camera angle with `Run YOLO every Nth frame = 2` and `Inference image size = 256` on CPU.
2. Observe **YOLO latency** and **Processing FPS** in the live metrics. If YOLO latency itself is very high, the detector/device is the bottleneck; changing CSS alone will not help.
3. Compare `stride = 1`, `2`, and `3` as well as image sizes `256`, `320`, and `416`. Higher stride means boxes are reused longer between detection steps; lower size may hurt detection of distant people.
4. The MJPEG viewer sends video independently of Streamlit's regular image-update path, reducing UI serialization overhead. It cannot guarantee 25/30 FPS and cannot make slow inference faster.
5. For compatible NVIDIA hardware, check that `torch.cuda.is_available()` is true and that an appropriate CUDA-enabled PyTorch build is installed. The current runtime auto-selects CUDA when available and otherwise uses CPU.

## Known limitations / demo claims

- This is a hackathon prototype, not certified access-control or biometric security software.
- The authorized log means a matched person was detected within the user-drawn rectangle; it does not prove that a person crossed a physical doorway.
- YuNet/SFace threshold `0.363` is a starting point, not a validated match threshold for every camera or population.
- The preview blur is a best-effort blur of the upper part of the person box; it is not guaranteed anonymization.
- Face embeddings are biometric data. A SHA-256 digest of embeddings does not make the source biometric data anonymous. Use consented test photos/footage and restrict access to logs and snapshots.
- The Streamlit roster is not persisted. The CSV access log is local. Cloud alerts may include an evidence crop; the authorized access table itself contains no images.
- SQLite is suitable for local testing. Use PostgreSQL for a deployed backend because ephemeral cloud filesystems may lose SQLite data.
- The backend reset route is unauthenticated and disabled by default. Do not enable it on a public deployment.

## Cloud deployment summary

- Deploy `backend/` to Render using `pip install -r requirements.txt` and `uvicorn main:app --host 0.0.0.0 --port $PORT`; set `DATABASE_URL`, `ALLOWED_ORIGINS`, and keep `ENABLE_DEMO_RESET=false`.
- Deploy `frontend/` to Vercel with root directory `frontend`; set `NEXT_PUBLIC_API_URL` to the Render API URL and `NEXT_PUBLIC_ENABLE_DEMO_RESET=false`.
- Add the exact Vercel origin to the backend's `ALLOWED_ORIGINS`.
- Run the edge app locally where the webcam/video source is connected; set its API endpoint to the deployed backend URL to send incidents to the cloud.
