import os
import cv2
import time
import uuid
import sqlite3
import threading
from datetime import datetime
import json
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from services.scraper import fetch_weather
from services.chat_agent import respond_to_message

# =========================
# CONFIGURAÇÕES (Hardcoded para facilitar)
# =========================
CAMERA_SOURCE = 0  # 0 ativa a tua WebCam local
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
SAVE_DIR = "static/captures"
DB_PATH = "detections.db"
DEFAULT_LAT = -23.55
DEFAULT_LON = -46.63

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
            image_path TEXT,
            weather TEXT
        )
    """)
    # If DB existed without `weather`, try to add column (ignore if exists)
    try:
        cur.execute("ALTER TABLE events ADD COLUMN weather TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

def save_event(event_id: str, label: str, confidence: float, image_path: str, weather: str = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", 
               (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence, image_path, weather))
    conn.commit()
    conn.close()

def list_events(limit: int = 15):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, event_time, label, confidence, image_path, weather FROM events ORDER BY event_time DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        weather = None
        if r[5]:
            try:
                weather = json.loads(r[5])
            except Exception:
                weather = None
        out.append({"id": r[0], "event_time": r[1], "label": r[2], "confidence": r[3], "image_path": r[4], "weather": weather})
    return out

# =========================
# PROCESSAMENTO DE VÍDEO (YOLO)
# =========================
def process_stream():
    global last_frame
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    while True:
        try:
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
                    # fetch weather for event (best-effort)
                    try:
                        weather_data = fetch_weather(DEFAULT_LAT, DEFAULT_LON)
                        weather_json = json.dumps(weather_data) if weather_data else None
                    except Exception:
                        weather_json = None
                    save_event(event_id, label, 0.9, f"/static/captures/{filename}", weather=weather_json)
                    last_alert_time[label] = time.time()
    
            with last_frame_lock:
                last_frame = frame.copy()
            time.sleep(0.03)
        except Exception as e:
            # log exception to a local file for debugging and continue
            try:
                with open('stream_errors.log', 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - process_stream error: {e}\n")
            except Exception:
                pass
            time.sleep(1)

@app.on_event("startup")
def startup_event():
    init_db()
    t = threading.Thread(target=process_stream, daemon=True)
    t.start()
    # store thread reference for debug
    global stream_thread
    stream_thread = t

@app.get('/api/debug')
def api_debug():
    """Debug endpoint: returns whether stream thread is alive and if a last_frame exists."""
    frame_exists = False
    try:
        with last_frame_lock:
            frame_exists = last_frame is not None
    except Exception:
        frame_exists = False
    thread_alive = False
    try:
        thread_alive = 'stream_thread' in globals() and stream_thread.is_alive()
    except Exception:
        thread_alive = False
    return JSONResponse(content={"frame_exists": frame_exists, "thread_alive": thread_alive})

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


@app.get("/api/weather")
def api_weather(lat: float, lon: float):
    """Retorna dados meteorológicos públicos (Open-Meteo) para as coordenadas fornecidas.

    Exemplo: `/api/weather?lat=-23.55&lon=-46.63`
    """
    data = fetch_weather(lat, lon)
    if data is None:
        return JSONResponse(status_code=503, content={"error": "source_unavailable"})
    return JSONResponse(content=data)


@app.post("/api/chat")
def api_chat(payload: dict):
    """POST /api/chat
    JSON body: { "message": "...", "event_id": "optional" }
    """
    message = payload.get('message') if isinstance(payload, dict) else None
    if not message:
        return JSONResponse(status_code=400, content={"error": "missing message"})

    event = None
    event_id = payload.get('event_id') if isinstance(payload, dict) else None
    if not event_id:
        # if no event_id provided, try to load the most recent event from the DB
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT id, event_time, label, confidence, image_path, weather FROM events ORDER BY event_time DESC LIMIT 1")
            r = cur.fetchone()
            conn.close()
            if r:
                weather = None
                try:
                    weather = json.loads(r[5]) if r[5] else None
                except Exception:
                    weather = None
                event = {"id": r[0], "event_time": r[1], "label": r[2], "confidence": r[3], "image_path": r[4], "weather": weather}
        except Exception:
            event = None
    else:
        # try to load event from DB
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT id, event_time, label, confidence, image_path, weather FROM events WHERE id = ?", (event_id,))
            r = cur.fetchone()
            conn.close()
            if r:
                weather = None
                try:
                    weather = json.loads(r[5]) if r[5] else None
                except Exception:
                    weather = None
                event = {"id": r[0], "event_time": r[1], "label": r[2], "confidence": r[3], "image_path": r[4], "weather": weather}
        except Exception:
            event = None

    resp = respond_to_message(message, event)
    return JSONResponse(content=resp)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/api/chat/provider")
def api_chat_provider():
    # Report 'llama' only if a model exists in LLAMA_MODEL_PATH or ./models/
    model_path = os.environ.get('LLAMA_MODEL_PATH')
    if not model_path:
        try:
            for fn in os.listdir('models'):
                if fn.lower().endswith(('.bin', '.ggml', '.ggml.bin')):
                    model_path = os.path.join('models', fn)
                    break
        except Exception:
            model_path = None

    provider = 'llama' if model_path else 'none'
    return JSONResponse(content={"provider": provider, "model_path": model_path})