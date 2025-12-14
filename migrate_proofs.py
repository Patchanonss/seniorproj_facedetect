import sqlite3
import os

DB_NAME = "attendance.db"

def migrate():
    print("📦 Starting Migration: Adding 'proof_path' to attendance_logs...")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(attendance_logs)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "proof_path" not in columns:
            print("   -> Column 'proof_path' missing. Adding it...")
            cursor.execute("ALTER TABLE attendance_logs ADD COLUMN proof_path TEXT")
            conn.commit()
            print("✅ Column added successfully.")
        else:
            print("✅ Column 'proof_path' already exists. Skipping.")
            
    except Exception as e:
        print(f"❌ Migration Failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
