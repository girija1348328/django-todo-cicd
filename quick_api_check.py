#!/usr/bin/env python3
"""
Quick check to see if detection history API is working.
"""

import requests
import sys

def check_api():
    base_url = "http://localhost:5000"
    
    print("🔍 Quick API Check")
    print("=" * 30)
    
    # Check if Flask app is running
    try:
        response = requests.get(f"{base_url}/", timeout=3)
        print(f"✅ Flask app is running (Status: {response.status_code})")
    except:
        print("❌ Flask app is NOT running")
        print("   Start it with: python app.py")
        return False
    
    # Check detection history endpoint
    try:
        response = requests.get(f"{base_url}/api/detection_history", timeout=5)
        print(f"✅ Detection History API: Status {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Retrieved {len(data)} records")
            if data:
                print(f"   📋 First record: {data[0]}")
        else:
            print(f"   ❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Detection History API failed: {e}")
        print("   This usually means the routes aren't registered")
        print("   Make sure to restart Flask app after adding routes")
    
    # Check if routes are registered
    try:
        response = requests.get(f"{base_url}/api/detection_history/stats", timeout=5)
        print(f"✅ Stats API: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Stats API failed: {e}")
    
    print("\n💡 If APIs are failing:")
    print("1. Make sure Flask app is restarted")
    print("2. Check that detection_history routes are registered")
    print("3. Verify the blueprint is imported in __init__.py")

if __name__ == "__main__":
    check_api()
