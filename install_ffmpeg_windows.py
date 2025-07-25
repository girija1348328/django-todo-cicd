#!/usr/bin/env python3
"""
FFMPEG Installation and Fix Script for Windows
This script helps install FFMPEG and fix OpenCV FFMPEG support issues.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_opencv_ffmpeg():
    """Check current OpenCV FFMPEG support"""
    try:
        import cv2
        build_info = cv2.getBuildInformation()
        has_ffmpeg = 'FFMPEG: YES' in build_info
        print(f"OpenCV version: {cv2.__version__}")
        print(f"FFMPEG support: {'✅ YES' if has_ffmpeg else '❌ NO'}")
        return has_ffmpeg
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def install_ffmpeg_windows():
    """Install FFMPEG on Windows"""
    print("\n🔧 Installing FFMPEG on Windows...")
    
    # Create temp directory
    temp_dir = Path("temp_ffmpeg")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download FFMPEG
        ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = temp_dir / "ffmpeg.zip"
        
        print("📥 Downloading FFMPEG...")
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        
        # Extract FFMPEG
        print("📦 Extracting FFMPEG...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the extracted directory
        extracted_dir = None
        for item in temp_dir.iterdir():
            if item.is_dir() and item.name.startswith("ffmpeg"):
                extracted_dir = item
                break
        
        if not extracted_dir:
            print("❌ Could not find extracted FFMPEG directory")
            return False
        
        # Copy FFMPEG to system PATH or local directory
        ffmpeg_dest = Path("ffmpeg")
        if ffmpeg_dest.exists():
            shutil.rmtree(ffmpeg_dest)
        
        shutil.copytree(extracted_dir, ffmpeg_dest)
        
        # Add to PATH for current session
        bin_path = ffmpeg_dest / "bin"
        current_path = os.environ.get('PATH', '')
        if str(bin_path) not in current_path:
            os.environ['PATH'] = f"{bin_path};{current_path}"
        
        print(f"✅ FFMPEG installed to: {ffmpeg_dest}")
        print("📝 Note: Add the following to your system PATH for permanent access:")
        print(f"   {bin_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error installing FFMPEG: {e}")
        return False
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def install_opencv_with_ffmpeg():
    """Install OpenCV with FFMPEG support"""
    print("\n🔧 Installing OpenCV with FFMPEG support...")
    
    try:
        # Uninstall current opencv-python
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "-y"], 
                      check=True, capture_output=True)
        
        # Install opencv-python-headless (includes FFMPEG)
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.1.78"], 
                      check=True)
        
        # Install ffmpeg-python for additional functionality
        subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python==0.2.0"], 
                      check=True)
        
        print("✅ OpenCV with FFMPEG support installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing OpenCV: {e}")
        return False

def test_ffmpeg_installation():
    """Test FFMPEG installation"""
    print("\n🧪 Testing FFMPEG installation...")
    
    try:
        # Test system FFMPEG
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ System FFMPEG is working")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Test local FFMPEG
    try:
        local_ffmpeg = Path("ffmpeg/bin/ffmpeg.exe")
        if local_ffmpeg.exists():
            result = subprocess.run([str(local_ffmpeg), "-version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Local FFMPEG is working")
                return True
    except subprocess.TimeoutExpired:
        pass
    
    print("❌ FFMPEG not working")
    return False

def test_opencv_rtsp():
    """Test OpenCV RTSP functionality"""
    print("\n🧪 Testing OpenCV RTSP support...")
    
    try:
        import cv2
        
        # Test RTSP URL parsing
        test_url = "rtsp://test:test@192.168.1.100:554/stream"
        cap = cv2.VideoCapture(test_url)
        
        # This will fail to connect, but we can check if FFMPEG is available
        build_info = cv2.getBuildInformation()
        
        if 'FFMPEG: YES' in build_info:
            print("✅ OpenCV has FFMPEG support")
            print("✅ RTSP streams should work")
            return True
        else:
            print("❌ OpenCV does not have FFMPEG support")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenCV RTSP: {e}")
        return False

def create_ffmpeg_config():
    """Create FFMPEG configuration for the application"""
    print("\n⚙️ Creating FFMPEG configuration...")
    
    config_content = '''# FFMPEG Configuration for Criminal Face Detection
# This file contains FFMPEG settings for RTSP streams

# RTSP Transport Protocol (tcp/udp)
RTSP_TRANSPORT = "tcp"

# Connection timeout in seconds
CONNECTION_TIMEOUT = 10

# Retry attempts for failed connections
MAX_RETRIES = 5

# Buffer size for video streams
BUFFER_SIZE = 1024 * 1024  # 1MB

# FFMPEG options for better compatibility
FFMPEG_OPTIONS = {
    "rtsp_transport": "tcp",
    "stimeout": "5000000",  # 5 seconds timeout
    "fflags": "nobuffer",
    "flags": "low_delay"
}

# Camera feed types and their FFMPEG settings
CAMERA_TYPES = {
    "rtsp": {
        "transport": "tcp",
        "timeout": 10,
        "retries": 5
    },
    "rtmp": {
        "transport": "tcp", 
        "timeout": 15,
        "retries": 3
    },
    "ip": {
        "transport": "tcp",
        "timeout": 5,
        "retries": 2
    }
}
'''
    
    with open("ffmpeg_config.py", "w") as f:
        f.write(config_content)
    
    print("✅ FFMPEG configuration created: ffmpeg_config.py")

def main():
    """Main installation function"""
    print("🚀 FFMPEG Installation and Fix Script for Criminal Face Detection")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check current OpenCV FFMPEG support
    print("\n📊 Current Status:")
    has_ffmpeg = check_opencv_ffmpeg()
    
    if has_ffmpeg:
        print("\n✅ FFMPEG support is already available!")
        return
    
    # Install FFMPEG
    if not install_ffmpeg_windows():
        print("\n❌ Failed to install FFMPEG")
        return
    
    # Install OpenCV with FFMPEG support
    if not install_opencv_with_ffmpeg():
        print("\n❌ Failed to install OpenCV with FFMPEG")
        return
    
    # Test installations
    print("\n🧪 Testing installations...")
    ffmpeg_ok = test_ffmpeg_installation()
    opencv_ok = test_opencv_rtsp()
    
    if ffmpeg_ok and opencv_ok:
        print("\n✅ All installations successful!")
        
        # Create configuration
        create_ffmpeg_config()
        
        print("\n🎯 Next Steps:")
        print("1. Restart your Python environment")
        print("2. Run your application: python app.py")
        print("3. Test RTSP camera feeds")
        print("4. If needed, add ffmpeg/bin to your system PATH")
        
    else:
        print("\n❌ Some installations failed")
        if not ffmpeg_ok:
            print("- FFMPEG installation failed")
        if not opencv_ok:
            print("- OpenCV FFMPEG support failed")

if __name__ == "__main__":
    main() 