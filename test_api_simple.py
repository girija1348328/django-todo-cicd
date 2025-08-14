#!/usr/bin/env python3
"""
Simple test to check if detection history API is working.
"""

import requests

def test_api():
    base_url = "http://localhost:5000"
    
    print("🔍 Testing Detection History API")
    print("=" * 40)
    
    # Test the API endpoint
    try:
        response = requests.get(f"{base_url}/api/detection_history", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Retrieved {len(data)} records")
            if data:
                print(f"📋 First record: {data[0]}")
        elif response.status_code == 404:
            print("❌ 404 - Endpoint not found")
            print("   The detection_history routes are not registered")
            print("   Make sure to restart Flask app after fixing routes")
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure Flask app is running")

if __name__ == "__main__":
    test_api()
