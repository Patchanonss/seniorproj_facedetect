
import sqlite3
import os
import sys

DB_NAME = "attendance.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def delete_student(identifier):
    conn = get_db_connection()
    try:
        # 1. Find the student
        student = conn.execute(
            "SELECT * FROM students WHERE name = ? OR student_code = ?", 
            (identifier, identifier)
        ).fetchone()

        if not student:
            print(f"❌ Student '{identifier}' not found in database.")
            return

        print(f"✅ Found student: {student['name']} (Code: {student['student_code']})")
        
        # 2. Get ID and Image Path
        sid = student['id']
        img_path = student['image_path']
        
        confirmation = input("⚠️  Are you sure you want to DELETE this student and all their logs? (y/N): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return

        # 3. Delete from Tables
        # Delete Enrollments
        conn.execute("DELETE FROM enrollments WHERE student_id = ?", (sid,))
        # Delete Logs
        conn.execute("DELETE FROM attendance_logs WHERE student_id = ?", (sid,))
        conn.execute("DELETE FROM raw_face_logs WHERE student_id = ?", (sid,))
        # Delete Student
        conn.execute("DELETE FROM students WHERE id = ?", (sid,))
        
        conn.commit()
        print("✅ Database records deleted.")

        # 4. Delete File
        # Check standard location and gallery location
        files_to_check = [img_path]
        # Also check specific gallery cache
        name_no_ext = student['name'].replace(" ", "_") # Rough guess if transformed
        
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"🗑️  Deleted photo: {img_path}")
        else:
            print(f"⚠️  Photo file not found at {img_path}")
            
        print("Note: You may still need to manually check 'gallery/' if files were moved.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_student.py <name_or_student_code>")
    else:
        delete_student(sys.argv[1])
