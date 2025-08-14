"""
Script to create the detection_history table in the database.
Run this script to set up the new table for storing detection events.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_folder.extensions import db
from app_folder.models.detection_history import DetectionHistory
from app_folder import create_app

def create_detection_history_table():
    """Create the detection_history table"""
    try:
        app = create_app()
        
        with app.app_context():
            db.create_all()
            
            try:
                count = DetectionHistory.query.count()
                print(f"✅ Detection history table created successfully!")
                print(f"   Current record count: {count}")
                print(f"   Table name: {DetectionHistory.__tablename__}")
                
                print(f"\n📋 Table structure:")
                print(f"   - id: Primary key (Integer)")
                print(f"   - timestamp: DateTime (default: UTC now)")
                print(f"   - access_granted: Boolean (required)")
                print(f"   - employee_name: String(100)")
                print(f"   - designation: String(100)")
                print(f"   - department: String(100)")
                print(f"   - confidence_score: Float")
                print(f"   - location: String(100)")
                print(f"   - camera_name: String(100)")
                print(f"   - camera_id: String(50)")
                print(f"   - detection_type: String(50)")
                print(f"   - image_data: Text (optional)")
                print(f"   - employee_id: Integer (foreign key to employee.id)")
                
            except Exception as e:
                print(f"❌ Error verifying table creation: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Error creating detection history table: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating detection_history table...")
    print("=" * 50)
    
    success = create_detection_history_table()
    
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
    else:
        print("\n💥 Setup failed! Please check the error messages above.")
        sys.exit(1)
