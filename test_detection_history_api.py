#!/usr/bin/env python3
"""
Test script for the detection history API endpoints.
Run this after starting your Flask application to test the API.
"""

import requests
import json
from datetime import datetime

# Base URL for the API (adjust if your app runs on a different port)
BASE_URL = "http://localhost:5000"

def test_store_detection():
    """Test storing a new detection event"""
    print("🧪 Testing: Store Detection Event")
    print("-" * 40)
    
    # Sample detection data
    detection_data = {
        "access": False,
        "employee_name": "Test Visitor",
        "designation": "Visitor",
        "department": "External",
        "confidence": 0.85,
        "location": "Main Entrance",
        "camera_name": "Cam01",
        "camera_id": "cam_001",
        "detection_type": "face_detection",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detection_history",
            json=detection_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Success! Detection stored with ID: {result.get('detection_id')}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_get_detection_history():
    """Test retrieving detection history"""
    print("\n🧪 Testing: Get Detection History")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/api/detection_history?limit=10")
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', [])
            total_count = result.get('total_count', 0)
            
            print(f"✅ Success! Retrieved {len(detections)} detections (Total: {total_count})")
            
            if detections:
                print("\n📋 Sample detection records:")
                for i, detection in enumerate(detections[:3]):  # Show first 3
                    print(f"   {i+1}. {detection.get('employee_name')} - "
                          f"{'GRANTED' if detection.get('access_granted') else 'DENIED'} "
                          f"at {detection.get('location')}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_get_detection_stats():
    """Test getting detection statistics"""
    print("\n🧪 Testing: Get Detection Statistics")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/api/detection_history/stats?days=7")
        
        if response.status_code == 200:
            result = response.json()
            stats = result.get('stats', {})
            
            print(f"✅ Success! Statistics for last 7 days:")
            print(f"   Total detections: {stats.get('total_detections', 0)}")
            print(f"   Granted: {stats.get('granted_count', 0)} ({stats.get('granted_percentage', 0)}%)")
            print(f"   Denied: {stats.get('denied_count', 0)} ({stats.get('denied_percentage', 0)}%)")
            
            camera_breakdown = stats.get('camera_breakdown', {})
            if camera_breakdown:
                print(f"   Camera breakdown: {camera_breakdown}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_filtered_history():
    """Test getting filtered detection history"""
    print("\n🧪 Testing: Get Filtered Detection History")
    print("-" * 40)
    
    try:
        # Test filtering by access type
        response = requests.get(f"{BASE_URL}/api/detection_history?access=denied&limit=5")
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', [])
            
            print(f"✅ Success! Retrieved {len(detections)} denied detections")
            
            if detections:
                print("   Sample denied detections:")
                for detection in detections[:2]:
                    print(f"     - {detection.get('employee_name')} at {detection.get('location')}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Detection History API Endpoints")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("Make sure your Flask application is running!")
    print()
    
    tests = [
        test_store_detection,
        test_get_detection_history,
        test_get_detection_stats,
        test_filtered_history
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The detection history API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("\n💡 Next steps:")
    print("1. The detection history is now being stored automatically")
    print("2. Check your database to see the stored records")
    print("3. The history sidebar will load data from the database")

if __name__ == "__main__":
    main()
