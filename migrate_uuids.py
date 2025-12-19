import sqlite3
import uuid

DB_NAME = "attendance.db"

def migrate():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    tables = ["subjects", "sessions", "students"]
    
    for table in tables:
        print(f"Checking table: {table}")
        # Check if uuid column exists
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "uuid" not in columns:
            print(f"Adding uuid column to {table}...")
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN uuid TEXT")
            
            # Populate existing rows with UUIDs
            cursor.execute(f"SELECT id FROM {table}")
            rows = cursor.fetchall()
            for row in rows:
                new_uuid = str(uuid.uuid4())
                cursor.execute(f"UPDATE {table} SET uuid = ? WHERE id = ?", (new_uuid, row[0]))
            print(f"Updated {len(rows)} rows in {table}.")
        else:
            print(f"Table {table} already has uuid column.")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
