import cv2
import time
from multiprocessing import Process, Queue, Manager

camera_processes = {}
camera_queues = {}

TARGET_FPS = 30
FRAME_DURATION = 1.0 / TARGET_FPS


def camera_worker(camera_url, frame_queue, camera_id):
    cap = cv2.VideoCapture(camera_url)
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        if not frame_queue.full():
            frame_queue.put(frame)
        elapsed = time.time() - start_time
        sleep_time = max(0, FRAME_DURATION - elapsed)
        time.sleep(sleep_time)


def start_camera_process(camera_id, camera_url):
    if camera_id in camera_processes:
        return
    frame_queue = Queue(maxsize=2)
    p = Process(target=camera_worker, args=(camera_url, frame_queue, camera_id), daemon=True)
    p.start()
    camera_processes[camera_id] = p
    camera_queues[camera_id] = frame_queue


def stop_camera_process(camera_id):
    if camera_id in camera_processes:
        camera_processes[camera_id].terminate()
        camera_processes[camera_id].join()
        del camera_processes[camera_id]
        del camera_queues[camera_id]


def get_latest_frame(camera_id):
    if camera_id in camera_queues:
        try:
            while True:
                frame = camera_queues[camera_id].get_nowait()
        except Exception:
            pass
        return frame if 'frame' in locals() else None
    return None 