"""
Simple script to create the detection_history table directly in SQLite.
This avoids Flask app context issues and directly creates the table.
"""

import os
import sqlite3

def create_detection_history_table_simple():
    """Create the detection_history table directly in SQLite"""
    db_path = 'database.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at: {db_path}")
        print("   Creating new database file...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            access_granted BOOLEAN NOT NULL,
            employee_name VARCHAR(100),
            designation VARCHAR(100),
            department VARCHAR(100),
            confidence_score REAL,
            location VARCHAR(100),
            camera_name VARCHAR(100),
            camera_id VARCHAR(50),
            detection_type VARCHAR(50),
            image_data TEXT,
            employee_id INTEGER,
            FOREIGN KEY (employee_id) REFERENCES employee (id)
        )
        """
        
        cursor.execute(create_table_sql)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detection_history_timestamp 
        ON detection_history(timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detection_history_access 
        ON detection_history(access_granted)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detection_history_camera 
        ON detection_history(camera_name)
        """)
        
        conn.commit()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='detection_history'
        """)
        
        if cursor.fetchone():
            print("✅ Detection history table created successfully!")
            
            cursor.execute("PRAGMA table_info(detection_history)")
            columns = cursor.fetchall()
            
            print(f"\n📋 Table structure:")
            print("-" * 80)
            print(f"{'Column':<20} {'Type':<15} {'Nullable':<10} {'Default':<15}")
            print("-" * 80)
            
            for col in columns:
                nullable = "NO" if col[3] else "YES"
                default = str(col[4]) if col[4] else "NULL"
                print(f"{col[1]:<20} {col[2]:<15} {nullable:<10} {default:<15}")
            
            cursor.execute("SELECT COUNT(*) FROM detection_history")
            count = cursor.fetchone()[0]
            print(f"\n📊 Current record count: {count}")
            
            return True
        else:
            print("❌ Failed to create detection_history table")
            return False
            
    except Exception as e:
        print(f"❌ Error creating detection history table: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def verify_existing_tables():
    """Show existing tables in the database"""
    db_path = 'database.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\n📋 Existing tables in database:")
        print("-" * 40)
        for table in tables:
            print(f"  - {table[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking existing tables: {e}")

if __name__ == "__main__":
    print("🚀 Creating detection_history table (Simple Method)...")
    print("=" * 60)
    
    verify_existing_tables()
    
    success = create_detection_history_table_simple()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Restart your Flask application")
        print("2. The detection history API endpoints are now available:")
        print("   - POST /api/detection_history - Store new detection")
        print("   - GET /api/detection_history - Retrieve history")
        print("   - GET /api/detection_history/stats - Get statistics")
        print("   - DELETE /api/detection_history/<id> - Delete specific record")
        print("   - POST /api/detection_history/clear - Clear all history")
        print("\n💡 You can now run: python view_detection_history.py")
    else:
        print("\n💥 Setup failed! Please check the error messages above.")
        sys.exit(1)
