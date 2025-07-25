import cv2
import time
from multiprocessing import Process, Queue, Manager

# Dictionary to keep track of camera processes and their queues
camera_processes = {}
camera_queues = {}

# Camera worker function

def camera_worker(camera_url, frame_queue, camera_id):
    cap = cv2.VideoCapture(camera_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        # (Optional) Add face recognition/model inference here
        if not frame_queue.full():
            frame_queue.put(frame)
        # Limit FPS to avoid overloading CPU/GPU
        time.sleep(0.03)  # ~30 FPS

# Function to start a camera process

def start_camera_process(camera_id, camera_url):
    if camera_id in camera_processes:
        return  # Already running
    frame_queue = Queue(maxsize=2)
    p = Process(target=camera_worker, args=(camera_url, frame_queue, camera_id), daemon=True)
    p.start()
    camera_processes[camera_id] = p
    camera_queues[camera_id] = frame_queue

# Function to stop a camera process

def stop_camera_process(camera_id):
    if camera_id in camera_processes:
        camera_processes[camera_id].terminate()
        camera_processes[camera_id].join()
        del camera_processes[camera_id]
        del camera_queues[camera_id]

# Function to get the latest frame from a camera

def get_latest_frame(camera_id):
    if camera_id in camera_queues:
        try:
            while True:
                frame = camera_queues[camera_id].get_nowait()
        except Exception:
            pass
        return frame if 'frame' in locals() else None
    return None 