from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import cv2
import numpy as np
import os
import time

# --- CUSTOM MODULES ---
# import core_AI
import database as db
import monolith_core

# --- GLOBAL INSTANCE ---
# This initializes the PyTorch-based AI system
ai_system = monolith_core.AICameraSystem()
# ai_system = core_AI.AICameraSystem()


# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db.init_db()
    # Lazy Start: We DON'T start the loop here anymore.
    # ai_system.start_threads() is called by the endpoints.
    print("✅ API Server Started (AI Camera is Lazy-Loaded)")
    yield
    # Shutdown
    ai_system.stop_loop()
    print("🛑 API Server Stopped")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DATA MODELS ---
class ClassSession(BaseModel):
    topic: str


class ManualUpdate(BaseModel):
    student_name: str
    status: str  # "PRESENT", "LATE", "ABSENT"


class ConfirmPayload(BaseModel):
    token: str
    name: str
    student_code: str


# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "System is running"}


# 1. THE VIDEO STREAM
def generate_frames(clean=False):
    while True:
        if clean:
            frame_bytes = ai_system.get_clean_frame()
        else:
            frame_bytes = ai_system.get_latest_frame()
            
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)


@app.get("/video_feed")
def video_feed(clean: bool = False):
    # LAZY START: Only open camera when feed is requested
    if not ai_system.threads_started:
        ai_system.start_threads()
        
    return StreamingResponse(generate_frames(clean), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/clean")
def video_feed_clean():
    # LAZY START
    if not ai_system.threads_started:
        ai_system.start_threads()
        
    return StreamingResponse(generate_frames(clean=True), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/capture_frame")
def capture_frame_endpoint():
    # LAZY START
    if not ai_system.threads_started:
        ai_system.start_threads()
        
    # Wait briefly for a frame if just started
    for _ in range(20):
        if ai_system.current_clean_jpeg is not None:
            break
        time.sleep(0.1)
        
    if ai_system.current_clean_jpeg:
        return Response(content=ai_system.current_clean_jpeg, media_type="image/jpeg")
    return Response(status_code=503, content="Camera initializing...")


# 2. CONTROL THE CLASS
@app.post("/start_class")
def start_class(session: ClassSession):
    session_id = db.create_session(session.topic)
    ai_system.active_session_id = session_id
    return {"message": "Class started", "session_id": session_id}


@app.post("/end_class")
def end_class():
    current_session = ai_system.active_session_id
    if current_session:
        db.close_session(current_session)
    ai_system.active_session_id = None
    return {"message": "Class ended"}


# 3. DATA FOR DASHBOARD
@app.get("/attendance/live")
def get_live_attendance():
    """Fetches real-time logs for the Dashboard table"""
    if ai_system.active_session_id is None:
        return {"status": "inactive", "logs": []}

    logs = db.get_logs_by_session(ai_system.active_session_id)

    return {
        "status": "active",
        "session_id": ai_system.active_session_id,
        "logs": logs
    }


# 4. MANUAL OVERRIDE ENDPOINT
@app.post("/attendance/manual")
def manual_attendance(update: ManualUpdate):
    if ai_system.active_session_id is None:
        raise HTTPException(status_code=400, detail="No active class session.")

    success = db.manual_update_status(
        ai_system.active_session_id,
        update.student_name,
        update.status
    )

    if not success:
        raise HTTPException(status_code=404, detail="Student not found.")

    return {"message": f"Updated {update.student_name} to {update.status}"}


# 5. REGISTRATION ENDPOINT (UPDATED FOR PYTORCH)
@app.post("/register_student")
async def register_student(
        name: str = Form(...),
        student_code: str = Form(...),
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    # 1. READ IMAGE
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. VALIDATE FACE (Using YOLO instead of DeepFace)
    # This server-side check prevents saving bad images
    try:
        results = ai_system.detector.predict(img, verbose=False, conf=0.5, device='cpu')
        # Check if any face boxes were detected
        if not results or not results[0].boxes:
            raise ValueError("No face")
    except Exception:
        raise HTTPException(status_code=400, detail="No face detected. Please take a clearer photo.")

    # 3. SAVE TO GALLERY
    # Create clean filename (e.g. "Chanon_S.jpg")
    safe_name = "".join(x for x in name if x.isalnum() or x in " _-").strip()
    if not os.path.exists("gallery"):
        os.makedirs("gallery")

    file_path = f"gallery/{safe_name}.jpg"
    cv2.imwrite(file_path, img)

    # 4. SAVE TO DATABASE
    success = db.add_student(student_code, safe_name, file_path)
    if not success:
        # If DB fails (duplicate ID), delete the photo to avoid junk files
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="Student ID already exists.")

    # 5. RE-SYNC AI (Background Task)
    # This prevents the video feed from freezing while the AI re-learns
    print(f"🔄 Queued background indexing for {safe_name}...")
    background_tasks.add_task(ai_system.reload_database)

    return {"status": "success", "message": f"Student {safe_name} registered! AI is updating..."}


# 6. NEW STAGED REGISTRATION (Better Logic)
@app.post("/register/validate")
async def register_validate(file: UploadFile = File(...)):
    """
    Step 1: Upload photo. check quality. get token + preview.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    result = ai_system.validate_and_stage(img)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["reason"])
        
    return result


@app.post("/register/confirm")
async def register_confirm(payload: ConfirmPayload, background_tasks: BackgroundTasks):
    """
    Step 2: Commit payload. using token.
    """
    res = ai_system.commit_registration(payload.token, payload.name)
    
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
        
    file_path = res["path"]
    safe_name = res["name"]
    
    # Write to DB
    success = db.add_student(payload.student_code, safe_name, file_path)
    
    if not success:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="Student ID already exists.")
        
        raise HTTPException(status_code=400, detail="Student ID already exists.")
        
    # No need to reload whole DB, we Hot-Added it in memory!
    print(f"✅ Student {safe_name} committed to DB and Memory.")
    
    return {"status": "success", "message": f"Student {safe_name} registered successfully!"}