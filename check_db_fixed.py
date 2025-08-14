#!/usr/bin/env python3
"""
Fixed database checker that shows all existing tables and their contents.
This will help us understand your actual database structure.
"""

import sqlite3
import os

def check_database():
    db_path = 'database.db'
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        print(f"   Expected location: {os.path.abspath(db_path)}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"✅ Database found at: {os.path.abspath(db_path)}")
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            print("📭 No tables found in database")
            return
        
        print(f"\n📋 Found {len(tables)} table(s):")
        print("-" * 40)
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check each table for content
        for table_name in [table[0] for table in tables]:
            print(f"\n🔍 Table: {table_name}")
            print("-" * 40)
            
            try:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  Records: {count}")
                
                if count > 0:
                    # Get table structure
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    print(f"  Columns: {len(columns)}")
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2]
                        nullable = "NOT NULL" if col[3] else "NULL"
                        print(f"    - {col_name}: {col_type} ({nullable})")
                    
                    # Show sample data (first 3 rows)
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = cursor.fetchall()
                    
                    if sample_rows:
                        print(f"  Sample data (first {len(sample_rows)} rows):")
                        for i, row in enumerate(sample_rows, 1):
                            print(f"    Row {i}: {row}")
                
            except Exception as e:
                print(f"  ❌ Error reading table: {e}")
        
        # Check for detection_history table specifically
        print(f"\n🎯 Detection History Status:")
        print("-" * 40)
        
        if 'detection_history' in [table[0] for table in tables]:
            cursor.execute("SELECT COUNT(*) FROM detection_history")
            count = cursor.fetchone()[0]
            print(f"✅ Detection history table exists with {count} records")
            
            if count > 0:
                cursor.execute("""
                    SELECT id, timestamp, access_granted, employee_name, location, camera_name
                    FROM detection_history 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                recent = cursor.fetchall()
                print(f"\n📋 Recent detection records:")
                print("-" * 80)
                print(f"{'ID':<4} {'Time':<20} {'Status':<8} {'Name':<20} {'Location':<15} {'Camera':<10}")
                print("-" * 80)
                
                for record in recent:
                    status = "✅ GRANTED" if record[2] else "❌ DENIED"
                    time_str = record[1][:19] if record[1] else "N/A"
                    print(f"{record[0]:<4} {time_str:<20} {status:<8} {record[3]:<20} {record[4]:<15} {record[5]:<10}")
            else:
                print("📭 Table is empty - no detection records yet")
        else:
            print("❌ Detection history table NOT found")
            print("   Run: python create_detection_history_table_simple.py")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

def check_flask_routes():
    """Check if Flask app has the detection history routes"""
    print(f"\n🔍 Flask App Route Check:")
    print("-" * 40)
    print("1. Make sure your Flask application is running")
    print("2. Make sure you restarted it after adding the detection_history routes")
    print("3. Test the API endpoint:")
    print("   curl http://localhost:5000/api/detection_history")
    print("\n4. If you get a 404 error, the routes aren't loaded")
    print("5. If you get a JSON response, the routes are working")

if __name__ == "__main__":
    print("🔍 Fixed Database Checker")
    print("=" * 50)
    check_database()
    check_flask_routes()
    print("\n" + "=" * 50)
    print("💡 Next steps:")
    print("1. If detection_history table is missing, run: python create_detection_history_table_simple.py")
    print("2. If table exists but is empty, check if Flask routes are working")
    print("3. If Flask routes work, check browser console for API call errors")
