#!/usr/bin/env python3
"""
Script to view detection history directly from the database.
This gives you a detailed view of all stored detection events.
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta

def connect_database():
    """Connect to the SQLite database"""
    db_path = 'database.db'
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at: {db_path}")
        print("   Make sure you're running this script from the project root directory.")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This allows accessing columns by name
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def check_table_exists(conn):
    """Check if the detection_history table exists"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='detection_history'
    """)
    return cursor.fetchone() is not None

def show_table_structure(conn):
    """Show the structure of the detection_history table"""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(detection_history)")
    columns = cursor.fetchall()
    
    print("\n📋 Table Structure:")
    print("-" * 80)
    print(f"{'Column':<20} {'Type':<15} {'Nullable':<10} {'Default':<15}")
    print("-" * 80)
    
    for col in columns:
        nullable = "NO" if col[3] else "YES"
        default = str(col[4]) if col[4] else "NULL"
        print(f"{col[1]:<20} {col[2]:<15} {nullable:<10} {default:<15}")

def show_recent_detections(conn, limit=20):
    """Show recent detection events"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, access_granted, employee_name, designation, 
               department, confidence_score, location, camera_name, camera_id
        FROM detection_history 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    detections = cursor.fetchall()
    
    if not detections:
        print("\n📭 No detection records found in the database.")
        print("   This means either:")
        print("   1. No detections have occurred yet")
        print("   2. The detection history system isn't storing data")
        print("   3. The table is empty")
        return
    
    print(f"\n📊 Recent Detection Events (Last {len(detections)}):")
    print("=" * 120)
    print(f"{'ID':<4} {'Time':<20} {'Status':<8} {'Name':<20} {'Location':<15} {'Camera':<10} {'Confidence':<10}")
    print("=" * 120)
    
    for detection in detections:
        status = "✅ GRANTED" if detection['access_granted'] else "❌ DENIED"
        time_str = detection['timestamp'][:19] if detection['timestamp'] else "N/A"
        confidence = f"{detection['confidence_score']:.2f}" if detection['confidence_score'] else "N/A"
        
        print(f"{detection['id']:<4} {time_str:<20} {status:<8} {detection['employee_name']:<20} "
              f"{detection['location']:<15} {detection['camera_name']:<10} {confidence:<10}")

def show_detection_stats(conn, days=7):
    """Show detection statistics"""
    cursor = conn.cursor()
    
    # Calculate date range
    date_from = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Total detections in date range
    cursor.execute("""
        SELECT COUNT(*) FROM detection_history 
        WHERE DATE(timestamp) >= ?
    """, (date_from,))
    total = cursor.fetchone()[0]
    
    # Granted vs Denied counts
    cursor.execute("""
        SELECT access_granted, COUNT(*) FROM detection_history 
        WHERE DATE(timestamp) >= ?
        GROUP BY access_granted
    """, (date_from,))
    
    granted = 0
    denied = 0
    for row in cursor.fetchall():
        if row[0]:
            granted = row[1]
        else:
            denied = row[1]
    
    # Camera breakdown
    cursor.execute("""
        SELECT camera_name, COUNT(*) FROM detection_history 
        WHERE DATE(timestamp) >= ?
        GROUP BY camera_name
        ORDER BY COUNT(*) DESC
    """, (date_from,))
    
    camera_stats = cursor.fetchall()
    
    print(f"\n📈 Detection Statistics (Last {days} days):")
    print("-" * 50)
    print(f"Total Detections: {total}")
    print(f"Access Granted: {granted} ({granted/total*100:.1f}%)" if total > 0 else "Access Granted: 0")
    print(f"Access Denied: {denied} ({denied/total*100:.1f}%)" if total > 0 else "Access Denied: 0")
    
    if camera_stats:
        print(f"\n📹 Camera Breakdown:")
        for camera, count in camera_stats:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {camera}: {count} ({percentage:.1f}%)")

def show_filtered_detections(conn, access_type=None, camera=None, limit=10):
    """Show filtered detection events"""
    cursor = conn.cursor()
    
    query = """
        SELECT id, timestamp, access_granted, employee_name, designation, 
               department, confidence_score, location, camera_name, camera_id
        FROM detection_history 
        WHERE 1=1
    """
    params = []
    
    if access_type:
        if access_type.lower() == 'granted':
            query += " AND access_granted = 1"
        elif access_type.lower() == 'denied':
            query += " AND access_granted = 0"
    
    if camera:
        query += " AND camera_name LIKE ?"
        params.append(f"%{camera}%")
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    detections = cursor.fetchall()
    
    if not detections:
        print(f"\n📭 No detections found with the specified filters.")
        return
    
    filter_desc = []
    if access_type:
        filter_desc.append(f"Access: {access_type}")
    if camera:
        filter_desc.append(f"Camera: {camera}")
    
    print(f"\n🔍 Filtered Detections ({', '.join(filter_desc)}):")
    print("=" * 120)
    print(f"{'ID':<4} {'Time':<20} {'Status':<8} {'Name':<20} {'Location':<15} {'Camera':<10} {'Confidence':<10}")
    print("=" * 120)
    
    for detection in detections:
        status = "✅ GRANTED" if detection['access_granted'] else "❌ DENIED"
        time_str = detection['timestamp'][:19] if detection['timestamp'] else "N/A"
        confidence = f"{detection['confidence_score']:.2f}" if detection['confidence_score'] else "N/A"
        
        print(f"{detection['id']:<4} {time_str:<20} {status:<8} {detection['employee_name']:<20} "
              f"{detection['location']:<15} {detection['camera_name']:<10} {confidence:<10}")

def show_employee_patterns(conn):
    """Show detection patterns by employee"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT employee_name, 
               COUNT(*) as total_detections,
               SUM(CASE WHEN access_granted = 1 THEN 1 ELSE 0 END) as granted,
               SUM(CASE WHEN access_granted = 0 THEN 1 ELSE 0 END) as denied
        FROM detection_history 
        WHERE employee_name != 'Unknown Visitor'
        GROUP BY employee_name
        ORDER BY total_detections DESC
        LIMIT 20
    """)
    
    patterns = cursor.fetchall()
    
    if not patterns:
        print("\n📭 No employee detection patterns found.")
        return
    
    print(f"\n👥 Employee Detection Patterns:")
    print("=" * 80)
    print(f"{'Name':<25} {'Total':<8} {'Granted':<10} {'Denied':<8} {'Success Rate':<12}")
    print("=" * 80)
    
    for pattern in patterns:
        success_rate = (pattern['granted'] / pattern['total_detections'] * 100) if pattern['total_detections'] > 0 else 0
        print(f"{pattern['employee_name']:<25} {pattern['total_detections']:<8} {pattern['granted']:<10} "
              f"{pattern['denied']:<8} {success_rate:<11.1f}%")

def main():
    """Main function to display detection history"""
    print("🔍 Detection History Database Viewer")
    print("=" * 50)
    
    # Connect to database
    conn = connect_database()
    if not conn:
        return
    
    try:
        # Check if table exists
        if not check_table_exists(conn):
            print("❌ The 'detection_history' table does not exist!")
            print("\n💡 To create the table, run:")
            print("   python create_detection_history_table.py")
            return
        
        print("✅ Connected to database successfully!")
        print("✅ Detection history table found!")
        
        # Show table structure
        show_table_structure(conn)
        
        # Show recent detections
        show_recent_detections(conn, limit=20)
        
        # Show statistics
        show_detection_stats(conn, days=7)
        
        # Show employee patterns
        show_employee_patterns(conn)
        
        # Interactive filtering
        print(f"\n🔍 Interactive Filtering Options:")
        print("1. View all recent detections (default)")
        print("2. Filter by access type (granted/denied)")
        print("3. Filter by camera")
        print("4. View detailed employee patterns")
        
        choice = input("\nEnter your choice (1-4) or press Enter for default view: ").strip()
        
        if choice == "2":
            access_type = input("Enter access type (granted/denied): ").strip()
            show_filtered_detections(conn, access_type=access_type, limit=15)
        elif choice == "3":
            camera = input("Enter camera name (or part of name): ").strip()
            show_filtered_detections(conn, camera=camera, limit=15)
        elif choice == "4":
            show_employee_patterns(conn)
        else:
            print("\n📊 Showing default view (recent detections)")
            show_recent_detections(conn, limit=30)
        
        print(f"\n💡 Tips:")
        print("- Run this script anytime to see updated data")
        print("- The web interface automatically shows this data in real-time")
        print("- Use the API endpoints for programmatic access")
        print("- Check the logs for any errors in data storage")
        
    except Exception as e:
        print(f"❌ Error viewing detection history: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
