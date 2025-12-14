from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import cv2
import numpy as np
import os
import time
import io

# --- CUSTOM MODULES ---
# import core_AI
import database as db
import monolith_core
import auth

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
    # Allow any origin (localhost, 192.168.x.x, etc.)
    allow_origin_regex=r"http://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path for safety
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
PROOFS_DIR = os.path.join(ABS_PATH, "proofs")
if not os.path.exists(PROOFS_DIR):
    os.makedirs(PROOFS_DIR)

app.mount("/proofs", StaticFiles(directory=PROOFS_DIR), name="proofs")


# --- DATA MODELS ---
class ClassSession(BaseModel):
    topic: str
    subject_code: str = None  # Optional for legacy/soft
    room: str = "Default Room" # NEW

class SubjectPayload(BaseModel):
    code: str
    name: str

class EnrollmentPayload(BaseModel):
    student_code: str
    subject_code: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    full_name: str

class ManualUpdate(BaseModel):
    student_name: str
    status: str  # "PRESENT", "LATE", "ABSENT"

class ConfirmPayload(BaseModel):
    token: str
    name: str
    student_code: str
    class_id: int # NEW


# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "System is running"}


# --- AUTH ROUTES ---
@app.post("/auth/register")
def register(user: UserRegister):
    hashed_pw = auth.get_password_hash(user.password)
    success = db.create_professor(user.username, hashed_pw, user.full_name)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User created successfully"}

@app.post("/auth/login")
def login(user: UserLogin):
    prof = db.get_professor_by_username(user.username)
    if not prof or not auth.verify_password(user.password, prof['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = auth.create_access_token(data={"sub": prof['username']})
    return {"access_token": access_token, "token_type": "bearer", "user": {"id": prof['id'], "name": prof['full_name']}}

@app.get("/auth/me")
def read_users_me(current_user: dict = auth.Depends(auth.get_current_user)):
    return {"username": current_user['username'], "full_name": current_user['full_name'], "id": current_user['id']}


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
    
    # Return placeholder if no frame yet
    return Response(content=b"", media_type="image/jpeg")

@app.get("/session/monitor")
def get_monitor_data(session_id: int):
    # This endpoint is PUBLIC (or maybe protected by token, but kept simple for now)
    # Allows separate browser to view status.
    data = db.get_live_monitor_data(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


# 2. CONTROL THE CLASS
# 2. CONTROL THE CLASS
@app.post("/start_class")
def start_class(session: ClassSession, current_user: dict = auth.Depends(auth.get_current_user)):
    print(f"🚀 Starting Class: {session.topic} (Code: {session.subject_code})")
    
    # 1. Create Session in DB (Linked to Professor)
    session_id = db.create_session(
        topic=session.topic, 
        subject_code=session.subject_code,
        professor_id=current_user['id'], # LINK TO PROFESSOR
        room=session.room
    )
    
    # 2. Add Room Info to Session Object (Optional, for memory)
    # This part of the instruction was a comment, so the existing logic for subject_id and ai_system.load_class is retained.
    
    # 1. Determine Subject ID (Create if needed)
    subject_id = None
    if session.subject_code:
        subject_id = db.create_subject(session.subject_code, session.subject_code) # Lazy create name=code
        
    # 2. Create Session linked to Professor
    # This was the original call, now it's effectively the first step with the new comment.
    # session_id = db.create_session(
    #     topic=session.topic, 
    #     subject_code=session.subject_code, 
    #     professor_id=current_user['id'], 
    #     room=session.room
    # )
    
    # 3. Load AI Class
    if subject_id:
        print(f"🔄 Switching AI to Class ID: {subject_id}")
        ai_system.load_class(subject_id)
    else:
        print("⚠️ Starting Legacy/Open Session (No Class Isolation)")
        ai_system.load_class("legacy")

    ai_system.active_session_id = session_id
    return {"message": "Class started", "session_id": session_id}


@app.post("/end_class")
def end_class():
    current_session = ai_system.active_session_id
    if current_session:
        db.close_session(current_session)
    ai_system.active_session_id = None
    return {"message": "Class ended"}
    
@app.post("/session/toggle_registration")
def toggle_registration(enable: bool):
    ai_system.allow_registration = enable
    status = "Enabled" if enable else "Disabled"
    print(f"🔓 Registration {status}")
    return {"message": f"Registration {status}", "enabled": enable}

@app.get("/session/registration_status")
def get_registration_status():
    return {"enabled": ai_system.allow_registration}
    
    
# --- SUBJECT MANAGEMENT ---

@app.get("/subjects")
def get_subjects(current_user: dict = auth.Depends(auth.get_current_user)):
    conn = db.get_db_connection()
    # Filter by the logged-in professor's ID
    # Also join with enrollments to get student count
    query = '''
        SELECT s.*, COUNT(e.student_id) as student_count
        FROM subjects s
        LEFT JOIN enrollments e ON s.id = e.subject_id
        WHERE s.professor_id = ?
        GROUP BY s.id
    '''
    rows = conn.execute(query, (current_user['id'],)).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/subjects")
def create_subject_endpoint(payload: SubjectPayload, current_user: dict = auth.Depends(auth.get_current_user)):
    # Used by Manage Classes page
    conn = db.get_db_connection()
    try:
        # Check if code exists
        exist = conn.execute('SELECT id FROM subjects WHERE code = ?', (payload.code,)).fetchone()
        if exist:
             raise HTTPException(status_code=400, detail="Subject code already exists")
        
        conn.execute(
            'INSERT INTO subjects (code, name, professor_id) VALUES (?, ?, ?)',
            (payload.code, payload.name, current_user['id'])
        )
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))
    conn.close()
    return {"message": "Subject created"}

@app.post("/enroll")
def enroll_student(payload: EnrollmentPayload):
    success, msg = db.enroll_student(payload.student_code, payload.subject_code)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}


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

class StudentCheckIn(BaseModel):
    student_code: str

@app.post("/attendance/student_self_checkin")
def student_self_checkin(payload: StudentCheckIn):
    """
    Allows a student to manually check in using their Student ID (Code).
    Only works if a class session is active.
    """
    if ai_system.active_session_id is None:
        raise HTTPException(status_code=400, detail="No active class session.")
        
    conn = db.get_db_connection()
    try:
        # 1. Find Student
        student = conn.execute("SELECT * FROM students WHERE student_code = ?", (payload.student_code,)).fetchone()
        if not student:
            raise HTTPException(status_code=404, detail="Student ID not found.")
            
        student_id = student['id']
        name = student['name']
        
        # 2. Check Enrollment (Optional, but good practice)
        # For now, let's assume if they exist, they can try to check in.
        # But we should check if they are enrolled in the current session's subject.
        session = conn.execute("SELECT subject_id FROM sessions WHERE id = ?", (ai_system.active_session_id,)).fetchone()
        if session:
             enrollment = conn.execute(
                "SELECT 1 FROM enrollments WHERE student_id = ? AND subject_id = ?", 
                (student_id, session['subject_id'])
             ).fetchone()
             if not enrollment:
                 raise HTTPException(status_code=403, detail="You are not enrolled in this class.")

        # 3. Log Attendance (Status = PRESENT, is_manual = 1)
        # reusing manual_update_status logic but bypassing name lookup
        # actually db.manual_update_status takes name. Let's just do direct DB insert or use log_attendance and set manual flag?
        # log_attendance doesn't set manual flag.
        # manual_update_status uses name.
        # Let's just call manual_update_status since we have the name.
        
        success = db.manual_update_status(ai_system.active_session_id, name, "PRESENT")
        
        if success:
             return {"message": "Checked in successfully", "student_name": name}
        else:
             raise HTTPException(status_code=500, detail="Failed to log attendance.")
             
    finally:
        conn.close()
    
    
# 4.5 EXPORT ENDPOINT
# 4.5 EXPORT ENDPOINT
@app.get("/export/attendance")
def export_attendance(subject_id: int = None, professor_id: int = None):
    import pandas as pd
    from io import BytesIO
    
    # 1. Fetch ALL Data needed for the matrix (Explicit ID Filter)
    logs = db.get_full_semester_data(professor_id=professor_id, subject_id=subject_id)
    all_students = db.get_all_students(professor_id=professor_id, subject_id=subject_id)
    all_sessions = db.get_all_sessions(professor_id=professor_id, subject_id=subject_id)
    
    # 2. Prepare Lists for Keys
    # Students
    student_records = []
    for s in all_students:
        student_records.append({'student_code': s['student_code'], 'name': s['name']})
    student_df = pd.DataFrame(student_records).drop_duplicates(subset=['student_code'])
    
    # Sessions
    session_labels = []
    for s in all_sessions:
        # Use HH:MM to differentiate sessions on same day
        time_str = s['start_time'][:5] if s['start_time'] else "00:00"
        label = f"{s['date']} ({time_str}) - {s['topic']}"
        session_labels.append(label)
    # Deduplicate sessions just in case
    session_labels = sorted(list(set(session_labels)))

    # 3. Create Base DataFrame from Logs
    if not logs:
        # If no logs, likely no data OR empty class.
        # We still want the matrix if students exist.
        df = pd.DataFrame(columns=['student_code', 'name', 'session_label', 'status'])
    else:
        df = pd.DataFrame(logs)
        # Apply same formatting
        df['time_str'] = df['start_time'].apply(lambda x: x[:5] if x else "00:00")
        df['session_label'] = df['date'] + " (" + df['time_str'] + ") - " + df['topic']
    
    # 4. Pivot
    # Check if we have data to pivot
    if not df.empty:
        pivot_df = df.pivot_table(
            index=['student_code', 'name'], 
            columns='session_label', 
            values='status', 
            aggfunc='first'
        )
    else:
        pivot_df = pd.DataFrame()
        # Create empty structure if we have specific lists
        if not student_df.empty:
            pivot_df = pd.DataFrame(index=pd.MultiIndex.from_frame(student_df[['student_code', 'name']]))

    # 5. Reindex to Ensure ALL Students are visible (Even if absent)
    if not student_df.empty:
        student_index = pd.MultiIndex.from_frame(student_df[['student_code', 'name']])
        pivot_df = pivot_df.reindex(index=student_index)
    
    # Reindex Columns -> All Sessions
    if session_labels:
        pivot_df = pivot_df.reindex(columns=session_labels)
        
    # 6. Fill Gaps
    pivot_df = pivot_df.fillna('ABSENT')
    
    # 7. Export
    stream = io.StringIO()
    pivot_df.to_csv(stream)
    
    response = Response(content=stream.getvalue(), media_type="text/csv")
    filename = f"attendance_report_{subject_id}.csv" if subject_id else "attendance_report_all.csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    
    return response




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
    # 0. Check Gate
    if not ai_system.allow_registration:
        raise HTTPException(status_code=403, detail="Registration is currently closed by the Professor.")

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
    # Fix: User AI system's logical commit
    res = ai_system.commit_registration(payload.token, payload.name, payload.class_id)
    
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
        
    file_path = res["path"]
    safe_name = res["name"]
    
    # Write to DB
    # Note: DB add_student is global. Student table is global.
    # This is fine. Enrollments handle class membership.
    success = db.add_student(payload.student_code, safe_name, file_path)
    
    # AUTO ENROLL TO CLASS
    # payload.class_id is the database ID (int), so we pass it as subject_id
    db.enroll_student(payload.student_code, subject_id=payload.class_id)
    
    return {"status": "success", "message": f"Student {safe_name} registered successfully!"}