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

    # 1. STUDENTS TABLE
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        image_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2. SESSIONS TABLE
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        date TEXT,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        is_active BOOLEAN DEFAULT 1
    )
    ''')

    # 3. ATTENDANCE LOGS
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        student_id INTEGER,
        check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT,
        is_manual BOOLEAN DEFAULT 0,
        FOREIGN KEY (session_id) REFERENCES sessions (id),
        FOREIGN KEY (student_id) REFERENCES students (id),
        UNIQUE(session_id, student_id)
    )
    ''')

    conn.commit()
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


# --- STUDENT FUNCTIONS ---

def add_student(student_code, name, image_path):
    """Used by the Registration Page"""
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO students (student_code, name, image_path) VALUES (?, ?, ?)',
            (student_code, name, image_path)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"⚠️ Student {student_code} already exists.")
        return False
    finally:
        conn.close()


def get_student_by_name(name):
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE name = ?', (name,)).fetchone()
    conn.close()
    return student


# --- SESSION FUNCTIONS ---

def create_session(topic=None):
    conn = get_db_connection()
    now = datetime.datetime.now()

    if not topic:
        topic = f"Class - {now.strftime('%Y-%m-%d')}"

    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO sessions (topic, date, start_time, is_active) VALUES (?, ?, ?, 1)',
        (topic, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'))
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

def log_attendance(session_id, student_id, status="PRESENT"):
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO attendance_logs (session_id, student_id, status) VALUES (?, ?, ?)',
            (session_id, student_id, status)
        )
        conn.commit()
        print(f"✅ Logged Student ID {student_id}")
        return True
    except sqlite3.IntegrityError:
        return False  # Already logged
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

def get_full_semester_data():
    """
    Fetches every single log for the whole semester.
    Used to generate the Master Excel Sheet.
    """
    conn = get_db_connection()
    query = '''
        SELECT 
            s.student_code,
            s.name, 
            ses.date,
            ses.topic,
            l.status
        FROM students s
        JOIN attendance_logs l ON s.id = l.student_id
        JOIN sessions ses ON l.session_id = ses.id
        ORDER BY ses.date ASC, s.student_code ASC
    '''
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_students():
    """Need this to know who was ABSENT (didn't show up in logs)"""
    conn = get_db_connection()
    rows = conn.execute('SELECT student_code, name FROM students').fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_sessions():
    """Need this to create the columns in Excel"""
    conn = get_db_connection()
    rows = conn.execute('SELECT topic, date FROM sessions WHERE is_active = 0 ORDER BY date ASC').fetchall()
    conn.close()
    return [dict(row) for row in rows]
# Init on load
init_db()