import sqlite3
import os

def check_database():
    db_path = 'database.db'
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM criminal")
        criminal_count = cursor.fetchone()[0]
        print(f"✅ Criminals in database: {criminal_count}")
        
        if criminal_count == 0:
            print("⚠️  WARNING: No criminals in database!")
            print("   Detection won't work without criminals to detect.")
            print("   Please add some criminals first.")
        
        cursor.execute("SELECT COUNT(*) FROM camera_feed")
        camera_count = cursor.fetchone()[0]
        print(f"✅ Camera feeds in database: {camera_count}")
        
        if camera_count == 0:
            print("⚠️  WARNING: No camera feeds configured!")
            print("   Detection will use default webcam if available.")
        
        cursor.execute("SELECT COUNT(*) FROM detection_log")
        log_count = cursor.fetchone()[0]
        print(f"✅ Detection logs: {log_count}")
        
        cursor.execute("SELECT criminal_name, timestamp FROM detection_log ORDER BY timestamp DESC LIMIT 5")
        recent = cursor.fetchall()
        if recent:
            print("\n📋 Recent detections:")
            for name, timestamp in recent:
                print(f"   - {name} at {timestamp}")
        else:
            print("\n📋 No recent detections")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    print("🔍 Checking Criminal Face Recognition Database...")
    print("=" * 50)
    check_database()
    print("=" * 50)
    print("\n💡 Troubleshooting tips:")
    print("1. If no criminals: Add criminals via the web interface")
    print("2. If no camera feeds: Add camera feeds or use webcam")
    print("3. Check browser console for JavaScript errors")
    print("4. Ensure all dependencies are installed") 