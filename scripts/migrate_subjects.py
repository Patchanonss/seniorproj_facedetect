
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(subjects)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "is_registration_open" in columns:
            print("✅ Column 'is_registration_open' already exists in 'subjects' table.")
        else:
            print("⚠️ Column not found. Adding 'is_registration_open' to 'subjects'...")
            # Add column (Default 0 = Closed)
            cursor.execute("ALTER TABLE subjects ADD COLUMN is_registration_open BOOLEAN DEFAULT 0")
            conn.commit()
            print("✅ Migration Successful: Column added.")
            
    except Exception as e:
        print(f"❌ Migration Failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
