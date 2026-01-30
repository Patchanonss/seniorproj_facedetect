import sqlite3
import shutil

DB_NAME = "attendance.db"
BACKUP_NAME = "attendance_backup.db"

def migrate():
    # 0. Backup
    shutil.copyfile(DB_NAME, BACKUP_NAME)
    print(f"✅ Backup created: {BACKUP_NAME}")

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 1. Create NEW TABLE with Constraints
        # UNIQUE(subject_id, topic)
        # CHECK(topic = lower(topic))
        
        cursor.execute('''
        CREATE TABLE sessions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER,
            professor_id INTEGER,
            topic TEXT,
            room TEXT,
            date TEXT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            uuid TEXT UNIQUE,
            FOREIGN KEY (subject_id) REFERENCES subjects (id),
            FOREIGN KEY (professor_id) REFERENCES professors (id),
            UNIQUE(subject_id, topic),
            CHECK(topic = lower(topic))
        )
        ''')
        
        # 2. Migrate Data with Conflict Resolution
        cursor.execute("SELECT * FROM sessions")
        rows = cursor.fetchall()
        
        print(f"🔄 Migrating {len(rows)} sessions...")
        
        # Track seen (subject_id, topic) to handle duplicates
        seen_keys = set()
        
        for row in rows:
            data = dict(row)
            
            # A. Lowercase Topic
            original_topic = data['topic']
            lowered = original_topic.lower() if original_topic else "unknown"
            
            # B. Conflict Check (Scoped to Subject)
            subject_id = data['subject_id']
            base_topic = lowered
            counter = 1
            
            # If (subject_id, lowered) is already processed, rename it!
            # BUT we also need to check against the DB? No, we are building fresh.
            # Local set is enough if we process sequentially.
            
            final_topic = base_topic
            
            while (subject_id, final_topic) in seen_keys:
                counter += 1
                final_topic = f"{base_topic}-{counter}"
                
            seen_keys.add((subject_id, final_topic))
            
            if final_topic != original_topic:
                print(f"   ⚠️ Renamed: '{original_topic}' -> '{final_topic}' (Collision or Case)")

            # Insert into NEW table
            cursor.execute('''
                INSERT INTO sessions_new (
                    id, subject_id, professor_id, topic, room, date, 
                    start_time, end_time, is_active, uuid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['id'], data['subject_id'], data['professor_id'], final_topic, 
                data['room'], data['date'], data['start_time'], data['end_time'], 
                data['is_active'], data['uuid']
            ))

        # 3. Swap Tables
        cursor.execute("DROP TABLE sessions")
        cursor.execute("ALTER TABLE sessions_new RENAME TO sessions")
        
        conn.commit()
        print("✅ Migration Complete: 'sessions' table updated with lowercase & scoped unique constraints.")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration Failed: {e}")
        # Restore? Users can do manual restore from backup if needed
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
