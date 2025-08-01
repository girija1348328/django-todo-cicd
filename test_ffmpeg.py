#!/usr/bin/env python3
"""
Test script to check FFMPEG support in OpenCV
"""

import cv2

print("OpenCV version:", cv2.__version__)
print()

build_info = cv2.getBuildInformation()
print("FFMPEG check results:")
print("- 'FFMPEG: YES' in build_info:", 'FFMPEG: YES' in build_info)
print("- 'FFMPEG' in build_info:", 'FFMPEG' in build_info)

print("\nAll FFMPEG-related lines in build info:")
for line in build_info.split('\n'):
    if 'FFMPEG' in line:
        print(f"  {line}")

print("\nFull build info (first 1000 chars):")
print(build_info[:1000]) 