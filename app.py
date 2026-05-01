import os
import cv2
import time
import uuid
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# =========================
# CONFIGURAÇÕES (Hardcoded para facilitar)
# =========================
CAMERA_SOURCE = 0  # 0 ativa a tua WebCam local
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
SAVE_DIR = "static/captures"
DB_PATH = "detections.db"

TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}
MIN_CONSECUTIVE_FRAMES = 3
ALERT_COOLDOWN_SECONDS = 20

# =========================
# APP SETUP
# =========================
app = FastAPI(title="AgroVision AI")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO(MODEL_PATH)
last_frame = None
last_frame_lock = threading.Lock()

detection_state = defaultdict(int)
last_alert_time = defaultdict(lambda: 0.0)

# =========================
# BANCO DE DADOS
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            event_time TEXT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_event(event_id: str, label: str, confidence: float, image_path: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", 
               (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence, image_path))
    conn.commit()
    conn.close()

def list_events(limit: int = 15):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, event_time, label, confidence, image_path FROM events ORDER BY event_time DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "event_time": r[1], "label": r[2], "confidence": r[3], "image_path": r[4]} for r in rows]

# =========================
# PROCESSAMENTO DE VÍDEO (YOLO)
# =========================
def process_stream():
    global last_frame
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            continue

        # Corre o YOLO no frame atual
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        found_now = set()

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                label = model.names[int(box.cls[0].item())]
                
                if label in TARGET_CLASSES:
                    found_now.add(label)
                    # Desenha a caixa no vídeo
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lógica de alerta e salvamento
        for label in found_now:
            detection_state[label] += 1
            if detection_state[label] >= MIN_CONSECUTIVE_FRAMES and (time.time() - last_alert_time[label] > ALERT_COOLDOWN_SECONDS):
                event_id = str(uuid.uuid4())[:8]
                filename = f"cap_{event_id}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, frame)
                save_event(event_id, label, 0.9, f"/static/captures/{filename}")
                last_alert_time[label] = time.time()
        
        with last_frame_lock:
            last_frame = frame.copy()
        time.sleep(0.03)

@app.on_event("startup")
def startup_event():
    init_db()
    threading.Thread(target=process_stream, daemon=True).start()

# =========================
# ROTAS DO DASHBOARD
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "events": list_events()})

def generate_frames():
    while True:
        with last_frame_lock:
            if last_frame is not None:
                _, buffer = cv2.imencode(".jpg", last_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")