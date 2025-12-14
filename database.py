import sqlite3
import datetime
import os

DB_NAME = "attendance.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn



def init_db(gallery_path="gallery"):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. PROFESSORS TABLE (NEW)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS professors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2. STUDENTS TABLE
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        image_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3. SUBJECTS TABLE (NEW/MODIFIED)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        professor_id INTEGER,
        FOREIGN KEY (professor_id) REFERENCES professors (id)
    )
    ''')

    # 4. SESSIONS TABLE (MODIFIED)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        professor_id INTEGER,
        topic TEXT,
        room TEXT,
        date TEXT,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        FOREIGN KEY (professor_id) REFERENCES professors (id)
    )
    ''')

    # 5. ATTENDANCE LOGS (MODIFIED)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        student_id INTEGER,
        professor_id INTEGER,
        check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT,
        is_manual BOOLEAN DEFAULT 0,
        proof_path TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (id),
        FOREIGN KEY (student_id) REFERENCES students (id),
        FOREIGN KEY (professor_id) REFERENCES professors (id),
    UNIQUE(session_id, student_id)
    )
    ''')
    
    # 5.5 RAW FACE LOGS (For "Store Everything" Requirement)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS raw_face_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        student_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidence REAL,
        proof_path TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (id),
        FOREIGN KEY (student_id) REFERENCES students (id)
    )
    ''')
    
    # 6. ENROLLMENTS TABLE
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS enrollments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        subject_id INTEGER,
        FOREIGN KEY (student_id) REFERENCES students (id),
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        UNIQUE(student_id, subject_id)
    )
    ''')
    conn.close()
    print("✅ Database Initialized!")

    # Auto Sync on startup
    sync_gallery_to_db(gallery_path)


def sync_gallery_to_db(gallery_path):
    if not os.path.exists(gallery_path):
        os.makedirs(gallery_path)
        return

    conn = get_db_connection()
    # Find all jpg files in gallery (Assuming filename is 'Name.jpg')
    # Or if you use folders, adjust accordingly.
    # Based on your previous code, it seemed you scanned directories.
    # If you changed to single files (Name.jpg), we scan files:
    existing_files = [f.name for f in os.scandir(gallery_path) if f.name.endswith(('.jpg', '.png'))]

    count = 0
    for filename in existing_files:
        name = os.path.splitext(filename)[0]  # Remove .jpg
        # Check if exists
        exists = conn.execute('SELECT 1 FROM students WHERE name = ?', (name,)).fetchone()
        if not exists:
            try:
                # Use name as temp student_code
                conn.execute('INSERT INTO students (student_code, name, image_path) VALUES (?, ?, ?)',
                             (name, name, os.path.join(gallery_path, filename)))
                count += 1
            except sqlite3.IntegrityError:
                pass

    if count > 0:
        conn.commit()
        print(f"🔄 Auto-Synced {count} students from Gallery.")
    conn.close()

# --- PROFESSOR FUNCTIONS (NEW) ---

def create_professor(username, password_hash, full_name):
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO professors (username, password_hash, full_name) VALUES (?, ?, ?)',
            (username, password_hash, full_name)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_professor_by_username(username):
    conn = get_db_connection()
    prof = conn.execute('SELECT * FROM professors WHERE username = ?', (username,)).fetchone()
    conn.close()
    return prof

def get_professor_by_id(prof_id):
    conn = get_db_connection()
    prof = conn.execute('SELECT * FROM professors WHERE id = ?', (prof_id,)).fetchone()
    conn.close()
    return prof


# --- STUDENT FUNCTIONS ---

def add_student(student_code, name, image_path):
    """
    Used by the Registration Page.
    Handles Multi-Class Registration by allowing existing students.
    """
    conn = get_db_connection()
    try:
        # Try to Insert
        conn.execute(
            'INSERT INTO students (student_code, name, image_path) VALUES (?, ?, ?)',
            (student_code, name, image_path)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Student exists. This is expected for multi-class enrollment.
        # We might want to update the image path or name?
        # For now, just return True so flow continues to Enrollment.
        print(f"ℹ️ Student {student_code} already in DB. Proceeding to enrollment.")
        return True
    finally:
        conn.close()


def get_student_by_name(name):
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE name = ?', (name,)).fetchone()
    conn.close()
    return student

def get_student_for_session(name, session_id):
    """
    Finds a student by name who is enrolled in the CURRENT session's subject.
    Resolves ambiguity if multiple students have the same name (e.g. multi-class enrollment duplicates).
    """
    conn = get_db_connection()
    try:
        # 1. Get Session Subject
        session = conn.execute('SELECT subject_id FROM sessions WHERE id = ?', (session_id,)).fetchone()
        if not session: return None
        
        subject_id = session['subject_id']
        
        # 2. Find Student with Name AND Enrollment in Subject
        query = '''
            SELECT s.* 
            FROM students s
            JOIN enrollments e ON s.id = e.student_id
            WHERE s.name = ? AND e.subject_id = ?
        '''
        student = conn.execute(query, (name, subject_id)).fetchone()
        
        # Fallback: If not found enrolled, just get any student with that name (will likely fail enrollment check later, but safe)
        if not student:
            student = conn.execute('SELECT * FROM students WHERE name = ?', (name,)).fetchone()
            
        return student
    finally:
        conn.close()


# --- SUBJECT & ENROLLMENT FUNCTIONS ---

def create_subject(code, name, professor_id=None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO subjects (code, name, professor_id) VALUES (?, ?, ?)', (code, name, professor_id))
        subject_id = cursor.lastrowid
        conn.commit()
        return subject_id
    except sqlite3.IntegrityError:
        # Return existing ID
        row = conn.execute('SELECT id FROM subjects WHERE code = ?', (code,)).fetchone()
        return row['id']
    finally:
        conn.close()

def enroll_student(student_code, subject_code=None, subject_id=None):
    conn = get_db_connection()
    try:
        # Get Student
        stu = conn.execute('SELECT id FROM students WHERE student_code = ?', (student_code,)).fetchone()
        if not stu: return False, "Student not found"
        
        # Get Subject
        sid = subject_id
        if not sid and subject_code:
            sub = conn.execute('SELECT id FROM subjects WHERE code = ?', (subject_code,)).fetchone()
            if sub: sid = sub['id']
            
        if not sid:
            return False, "Subject not found"
            
        conn.execute('INSERT INTO enrollments (student_id, subject_id) VALUES (?, ?)', (stu['id'], sid))
        conn.commit()
        return True, "Enrolled"
    except sqlite3.IntegrityError:
        return True, "Already Enrolled"
    finally:
        conn.close()

def check_enrollment(student_id, session_id):
    """
    Checks if a student is enrolled in the subject of the given session.
    If session has NO subject (Legacy/Open), returns True.
    """
    conn = get_db_connection()
    
    # 1. Get Session's Subject
    session = conn.execute('SELECT subject_id FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if not session or not session['subject_id']:
        conn.close()
        return True # Open Session
        
    # 2. Check Enrollment
    enrollment = conn.execute(
        'SELECT 1 FROM enrollments WHERE student_id = ? AND subject_id = ?', 
        (student_id, session['subject_id'])
    ).fetchone()
    
    conn.close()
    return True if enrollment else False


# --- SESSION FUNCTIONS ---

def create_session(topic=None, subject_code=None, professor_id=None, room=None):
    conn = get_db_connection()
    now = datetime.datetime.now()
    
    subject_id = None
    if subject_code:
        # Auto-create subject if passing a code directly (Soft Launch)
        # TODO: Assign professor_id to subject too if new?
        subject_id = create_subject(subject_code, subject_code) 
        # Update subject owner if not set? Skip for now.

    if not topic:
        topic = f"Class - {now.strftime('%Y-%m-%d')}"

    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO sessions (subject_id, professor_id, topic, room, date, start_time, is_active) VALUES (?, ?, ?, ?, ?, ?, 1)',
        (subject_id, professor_id, topic, room, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'))
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_active_session():
    conn = get_db_connection()
    session = conn.execute(
        'SELECT * FROM sessions WHERE is_active = 1 ORDER BY id DESC LIMIT 1'
    ).fetchone()
    conn.close()
    return session


def close_session(session_id):
    conn = get_db_connection()
    now = datetime.datetime.now()
    conn.execute(
        'UPDATE sessions SET is_active = 0, end_time = ? WHERE id = ?',
        (now.strftime('%H:%M:%S'), session_id)
    )
    conn.commit()
    conn.close()


# --- LOGGING FUNCTIONS ---

def log_raw_face(session_id, student_id, confidence, proof_path=None):
    """
    Logs EVERY face detection event. No unique constraint.
    """
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO raw_face_logs (session_id, student_id, confidence, proof_path) VALUES (?, ?, ?, ?)',
            (session_id, student_id, confidence, proof_path)
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️ Failed to log raw face: {e}")
    finally:
        conn.close()

def log_attendance(session_id, student_id, status="PRESENT", proof_path=None):
    conn = get_db_connection()
    try:
        # 1. Get Professor ID from Session
        session = conn.execute('SELECT professor_id FROM sessions WHERE id = ?', (session_id,)).fetchone()
        prof_id = session['professor_id'] if session else None

        conn.execute(
            'INSERT INTO attendance_logs (session_id, student_id, professor_id, status, proof_path) VALUES (?, ?, ?, ?, ?)',
            (session_id, student_id, prof_id, status, proof_path)
        )
        conn.commit()
        print(f"✅ Logged Student ID {student_id} for Prof {prof_id} (Proof: {proof_path})")
        return True
    except sqlite3.IntegrityError:
        return False  # Already logged
    finally:
        conn.close()

def manual_update_status(session_id, student_name, status):
    """
    Manually overrides the student status.
    Sets is_manual = 1.
    """
    conn = get_db_connection()
    try:
        # 1. Resolve Student ID
        stu = conn.execute("SELECT id FROM students WHERE name = ?", (student_name,)).fetchone()
        if not stu: return False
        
        student_id = stu['id']
        
        # 2. Get Professor ID from Session
        session = conn.execute('SELECT professor_id FROM sessions WHERE id = ?', (session_id,)).fetchone()
        prof_id = session['professor_id'] if session else None

        # 3. Upsert (Insert or Update)
        # Check if log exists
        exists = conn.execute(
            "SELECT 1 FROM attendance_logs WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        ).fetchone()
        
        if exists:
            conn.execute(
                "UPDATE attendance_logs SET status = ?, is_manual = 1, professor_id = ? WHERE session_id = ? AND student_id = ?",
                (status, prof_id, session_id, student_id)
            )
        else:
             conn.execute(
                "INSERT INTO attendance_logs (session_id, student_id, professor_id, status, is_manual) VALUES (?, ?, ?, ?, 1)",
                (session_id, student_id, prof_id, status)
            )
            
        conn.commit()
        return True
    except Exception as e:
        print(f"❌ Manual Update Error: {e}")
        return False
    finally:
        conn.close()


def get_logs_by_session(session_id):
    conn = get_db_connection()
    query = '''
        SELECT s.name, s.student_code, l.check_in_time, l.status 
        FROM attendance_logs l
        JOIN students s ON l.student_id = s.id
        WHERE l.session_id = ?
        ORDER BY l.check_in_time DESC
    '''
    rows = conn.execute(query, (session_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- REPORTING FUNCTIONS ---

def get_full_semester_data(professor_id=None, subject_id=None):
    """
    Fetches every single log for the whole semester.
    Used to generate the Master Excel Sheet.
    Includes filtering by Professor and Subject for isolation.
    """
    conn = get_db_connection()
    
    # Base Query
    query = """
        SELECT 
            s.student_code,
            s.name, 
            ses.date,
            ses.topic,
            ses.start_time,
            CASE 
                WHEN l.is_manual = 1 THEN l.status || ' (Manual)'
                ELSE l.status 
            END as status
        FROM students s
        JOIN attendance_logs l ON s.id = l.student_id
        JOIN sessions ses ON l.session_id = ses.id
        WHERE 1=1
    """
    params = []
    
    # Filters
    if professor_id:
        query += " AND ses.professor_id = ?"
        params.append(professor_id)
        
    if subject_id:
        query += " AND ses.subject_id = ?"
        params.append(subject_id)
        
    query += " ORDER BY ses.date ASC, ses.start_time ASC, s.student_code ASC"
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_students(professor_id=None, subject_id=None):
    """
    Need this to know who was ABSENT (didn't show up in logs).
    If professor_id/subject_id provided, only gets enrolled students.
    """
    conn = get_db_connection()
    
    if subject_id:
        # Strict Subject Enrollment
        query = """
            SELECT s.student_code, s.name 
            FROM students s
            JOIN enrollments e ON s.id = e.student_id
            WHERE e.subject_id = ?
        """
        params = [subject_id]
        
    elif professor_id:
        # All students enrolled in ANY subject of this Professor
        query = """
            SELECT DISTINCT s.student_code, s.name 
            FROM students s
            JOIN enrollments e ON s.id = e.student_id
            JOIN subjects sub ON e.subject_id = sub.id
            WHERE sub.professor_id = ?
        """
        params = [professor_id]
    else:
        # Dangerous Global Fallback (Should be avoided in API)
        query = 'SELECT student_code, name FROM students'
        params = []
        
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_sessions(professor_id=None, subject_id=None):
    """
    Need this to create the columns in Excel.
    Filters to return only relevant sessions.
    """
    conn = get_db_connection()
    
    query = "SELECT topic, date, start_time FROM sessions WHERE 1=1"
    params = []
    
    if professor_id:
        query += " AND professor_id = ?"
        params.append(professor_id)
        
    if subject_id:
        query += " AND subject_id = ?"
        params.append(subject_id)
        
    query += " ORDER BY date ASC, start_time ASC"
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_live_monitor_data(session_id):
    """
    Returns a list of ALL enrolled students for the session's subject.
    Status is 'PRESENT' if they have a log in this session, otherwise 'ABSENT'.
    """
    conn = get_db_connection()
    
    # 1. Get Subject ID from Session
    sess = conn.execute('SELECT subject_id, professor_id, topic, room, start_time FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if not sess:
        conn.close()
        return None
        
    subject_id = sess['subject_id']
    
    # 2. Get All Enrolled Students + Left Join with Logs
    query = '''
        SELECT 
            s.student_code,
            s.name,
            s.image_path,
            l.check_in_time,
            l.proof_path,
            CASE 
                WHEN l.status IS NOT NULL THEN l.status
                ELSE 'ABSENT' 
            END as status
        FROM enrollments e
        JOIN students s ON e.student_id = s.id
        LEFT JOIN attendance_logs l ON l.student_id = s.id AND l.session_id = ?
        WHERE e.subject_id = ?
        ORDER BY s.student_code ASC
    '''
    
    rows = conn.execute(query, (session_id, subject_id)).fetchall()
    conn.close()
    
    students = [dict(row) for row in rows]
    
    return {
        "session_info": dict(sess),
        "students": students
    }

# Init on load
init_db()