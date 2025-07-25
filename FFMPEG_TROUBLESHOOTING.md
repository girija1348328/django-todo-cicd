# FFMPEG Troubleshooting Guide

## Overview
This guide helps resolve FFMPEG-related issues in the Criminal Face Detection project, particularly for RTSP camera streams.

## Current Issues Identified

### 1. OpenCV FFMPEG Support Missing
**Problem**: OpenCV is installed without FFMPEG support
**Symptoms**: 
- `FFMPEG in build info: False`
- RTSP streams fail to connect
- Error messages about unsupported protocols

**Root Cause**: Using `opencv-python` instead of `opencv-python-headless`

### 2. System FFMPEG Not Installed
**Problem**: FFMPEG is not installed on Windows system
**Symptoms**:
- `ffmpeg : The term 'ffmpeg' is not recognized`
- RTSP streams fail even with OpenCV FFMPEG support

## Solutions

### Quick Fix (Recommended)

1. **Run the automated fix script**:
   ```bash
   python install_ffmpeg_windows.py
   ```

2. **Restart your Python environment**

3. **Test the fix**:
   ```bash
   python -c "import cv2; print('FFMPEG support:', 'FFMPEG: YES' in cv2.getBuildInformation())"
   ```

### Manual Fix

#### Step 1: Fix OpenCV FFMPEG Support
```bash
# Uninstall current OpenCV
pip uninstall opencv-python -y

# Install OpenCV with FFMPEG support
pip install opencv-python-headless==4.8.1.78

# Install additional FFMPEG Python bindings
pip install ffmpeg-python==0.2.0
```

#### Step 2: Install System FFMPEG

**Option A: Using Chocolatey (Recommended)**
```bash
# Install Chocolatey first if not installed
# Then install FFMPEG
choco install ffmpeg
```

**Option B: Manual Installation**
1. Download FFMPEG from: https://ffmpeg.org/download.html#build-windows
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH

**Option C: Using the provided script**
```bash
python install_ffmpeg_windows.py
```

#### Step 3: Verify Installation
```bash
# Check OpenCV FFMPEG support
python -c "import cv2; print('FFMPEG support:', 'FFMPEG: YES' in cv2.getBuildInformation())"

# Check system FFMPEG
ffmpeg -version
```

## Testing RTSP Streams

### Test Camera Connection
1. Go to Camera Feeds page in the web interface
2. Add a test RTSP camera with URL: `rtsp://test:test@192.168.1.100:554/stream`
3. Click "Test Connection" button

### Test with Sample RTSP Stream
You can use these public test streams:
- `rtsp://demo:demo@ipvmdemo.dyndns.org:5541/onvif-media/media.amp`
- `rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4`

## Common Error Messages and Solutions

### Error: "OpenCV is not built with FFMPEG support"
**Solution**: Install `opencv-python-headless` instead of `opencv-python`

### Error: "ffmpeg not found"
**Solution**: Install FFMPEG on your system

### Error: "RTSP connect failed"
**Solutions**:
1. Check if camera URL is correct
2. Verify network connectivity
3. Try different RTSP transport protocols (TCP/UDP)
4. Check firewall settings

### Error: "Camera not available (RTSP connect failed)"
**Solutions**:
1. Verify FFMPEG support is enabled
2. Check camera credentials
3. Test with a known working RTSP stream
4. Check network connectivity to camera

## Configuration Options

### RTSP Transport Protocol
The application automatically adds `rtsp_transport=tcp` to RTSP URLs for better reliability.

### Connection Timeouts
Default timeout is 10 seconds with 5 retry attempts.

### Buffer Settings
Default buffer size is 1MB for video streams.

## Advanced Troubleshooting

### Check OpenCV Build Information
```python
import cv2
build_info = cv2.getBuildInformation()
print(build_info)
```

### Test FFMPEG Directly
```bash
# Test RTSP stream with FFMPEG
ffmpeg -i "rtsp://your-camera-url" -t 5 -f null -
```

### Enable Debug Logging
Add this to your application startup:
```python
import cv2
cv2.setLogLevel(0)  # Enable all debug messages
```

### Check Network Connectivity
```bash
# Test network connectivity to camera
ping your-camera-ip
telnet your-camera-ip 554
```

## Performance Optimization

### For Better RTSP Performance
1. Use TCP transport for RTSP
2. Increase buffer sizes for high-bandwidth streams
3. Reduce frame processing frequency
4. Use hardware acceleration if available

### Recommended Settings
```python
# In your camera configuration
rtsp_url += "?rtsp_transport=tcp&buffer_size=1048576"
```

## Support

If you continue to have issues:

1. **Check the logs**: Look for detailed error messages in the console
2. **Test with known working streams**: Use the provided test URLs
3. **Verify your setup**: Run the diagnostic script
4. **Check network**: Ensure firewall allows RTSP traffic (port 554)

## Files Modified

- `requirements.txt`: Updated OpenCV package
- `app.py`: Enhanced FFMPEG detection and error handling
- `install_ffmpeg_windows.py`: Automated installation script
- `ffmpeg_config.py`: Configuration file (created by script)

## Next Steps

After fixing FFMPEG issues:

1. Restart your application
2. Test with a local camera first
3. Then test with RTSP streams
4. Monitor the logs for any remaining issues
5. Configure your camera feeds in the web interface 