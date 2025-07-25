#!/usr/bin/env python3
"""
Test script to verify RTSP stream functionality
"""

import cv2
import time

def test_rtsp_support():
    """Test if RTSP streams can be opened"""
    print("🧪 Testing RTSP Stream Support")
    print("=" * 40)
    
    # Check OpenCV FFMPEG support
    build_info = cv2.getBuildInformation()
    ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]
    
    print(f"OpenCV version: {cv2.__version__}")
    print(f"FFMPEG support: {'✅ YES' if ffmpeg_support else '❌ NO'}")
    
    if not ffmpeg_support:
        print("❌ FFMPEG support not available. RTSP streams will not work.")
        return False
    
    # Test with a public RTSP stream
    test_urls = [
        "rtsp://demo:demo@ipvmdemo.dyndns.org:5541/onvif-media/media.amp",
        "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    ]
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n📹 Testing RTSP stream {i}: {url}")
        
        try:
            # Try to open the stream
            cap = cv2.VideoCapture(url)
            
            if cap.isOpened():
                print("✅ Stream opened successfully")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Frame read successfully - Size: {frame.shape}")
                else:
                    print("⚠️  Stream opened but no frame could be read")
                
                cap.release()
                return True
            else:
                print("❌ Failed to open stream")
                
        except Exception as e:
            print(f"❌ Error opening stream: {e}")
    
    print("\n⚠️  All test streams failed. This might be due to:")
    print("   - Network connectivity issues")
    print("   - Firewall blocking RTSP traffic")
    print("   - Test streams being unavailable")
    print("   - Need for system FFMPEG installation")
    
    return False

def test_local_camera():
    """Test local camera functionality"""
    print("\n📹 Testing Local Camera")
    print("=" * 30)
    
    for i in range(3):  # Test cameras 0-2
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {i} working - Frame size: {frame.shape}")
                    cap.release()
                    return True
                else:
                    print(f"⚠️  Camera {i} opened but no frame")
                    cap.release()
            else:
                print(f"❌ Camera {i} not available")
        except Exception as e:
            print(f"❌ Error with camera {i}: {e}")
    
    print("❌ No local cameras found")
    return False

def main():
    """Main test function"""
    print("🚀 RTSP and Camera Test for Criminal Face Detection")
    print("=" * 60)
    
    # Test RTSP support
    rtsp_ok = test_rtsp_support()
    
    # Test local camera
    local_ok = test_local_camera()
    
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"RTSP Support: {'✅ Working' if rtsp_ok else '❌ Not Working'}")
    print(f"Local Camera: {'✅ Working' if local_ok else '❌ Not Working'}")
    
    if rtsp_ok:
        print("\n🎉 RTSP streams should work in your application!")
    else:
        print("\n⚠️  RTSP streams may not work. Consider:")
        print("   1. Installing system FFMPEG")
        print("   2. Checking network connectivity")
        print("   3. Testing with your specific camera URLs")
    
    if local_ok:
        print("\n🎉 Local cameras should work in your application!")
    else:
        print("\n⚠️  Local cameras not detected. Check camera connections.")

if __name__ == "__main__":
    main() 