# import eventlet
# eventlet.monkey_patch()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, current_app
from flask_sqlalchemy import SQLAlchemy
import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, current_app
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import time
from datetime import datetime, timedelta
from threading import Thread
import threading
import queue
import urllib.parse
from insightface.app import FaceAnalysis
from flask_migrate import Migrate
import onnxruntime as ort
from collections import deque
import base64
from face_tracker import FaceTracker
import traceback
import tensorflow as tf
from threading import Event

# Define threshold parameters for easy configuration
FACE_DETECTION_CONFIDENCE = 0.95  # Confidence threshold for face detection
FACE_RECOGNITION_THRESHOLD = 0.6  # Threshold for face recognition matching
MIN_FACE_SIZE_PX = 60  # Minimum face size in pixels
MAX_YAW_ANGLE_DEG = 30  # Maximum face rotation angle in degrees
MIN_RECOGNITION_MARGIN = 0.1  # Minimum difference between top matches

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_recognition_system")

print("OpenCV version:", cv2.__version__)
build_info = cv2.getBuildInformation()
ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]
print("FFMPEG in build info:", ffmpeg_support)
print("FFMPEG backend will be used for RTSP streams")
print("[DEBUG] ONNX Runtime device:", ort.get_device())

# Initialize InsightFace FaceAnalysis for detection and recognition
try:
    logger.info("Initializing InsightFace FaceAnalysis...")
    available_providers = ort.get_available_providers()
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]
    arcface_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    arcface_app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
    logger.info(f"FaceAnalysis initialized with providers: {arcface_app.det_model.session.get_providers()}")
except Exception as e:
    logger.error(f"Failed to initialize FaceAnalysis: {e}")
    arcface_app = None

from app_folder.extensions import db
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['DETECTION_COOLDOWN_SECONDS'] = 30  # Time window to prevent duplicate detections

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SocketIO
from flask_socketio import SocketIO
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

@socketio.on('connect')
def handle_connect():
    print("[DEBUG] Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("[DEBUG] Client disconnected")

db.init_app(app)
migrate = Migrate(app, db)

from app_folder.models.employee import Employee, EmployeeImage
from app_folder.models.camera_feed import CameraFeed
from app_folder.models.attendance_log import AttendanceLog

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_rtsp_url(rtsp_url):
    # Only encode if credentials are present
    if 'rtsp://' in rtsp_url:
        try:
            prefix, rest = rtsp_url.split('://', 1)
            if '@' in rest:
                creds, path = rest.split('@', 1)
                if ':' in creds:
                    user, pwd = creds.split(':', 1)
                    user_enc = urllib.parse.quote(user)
                    pwd_enc = urllib.parse.quote(pwd)
                    return f"{prefix}://{user_enc}:{pwd_enc}@{path}"
        except Exception as e:
            print(f"[WARN] Could not encode RTSP credentials: {e}")
    return rtsp_url

# @app.route('/')
# def home():
#     return redirect('/live_detection_page')

# Add a new column to Employee for ArcFace embedding (if not already present)
# If using Alembic or migrations, this should be handled there. For now, add in-memory only for demonstration.
if not hasattr(Employee, 'arcface_embedding'):
    from sqlalchemy import PickleType
    Employee.arcface_embedding = db.Column(PickleType, nullable=True)

# Initialize ArcFace model globally
ctx_id = -1  # Use CPU first for testing
    # ...existing code...

def extract_arcface_embedding_from_crop(face_crop):
    """
    Use InsightFace FaceAnalysis to extract embedding from a face crop.
    """
    if arcface_app is None:
        logger.error("FaceAnalysis not initialized.")
        return None
    try:
        crop = cv2.resize(face_crop, (112, 112))
        faces = arcface_app.get(crop)
        if not faces:
            logger.info("No face detected in crop.")
            return None
        embedding = faces[0].embedding
        if embedding is None or np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.info("Invalid embedding values.")
            return None
        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.info("Zero norm embedding.")
            return None
        embedding = embedding / norm
        logger.info(f"Valid embedding generated, shape: {embedding.shape}")
        return embedding
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None

def align_face_by_keypoints(img, keypoints, output_size=(112, 112)):
    # Standard ArcFace reference points for 112x112
    ref_pts = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041]    # right mouth
    ], dtype=np.float32)

    src_pts = np.array([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ], dtype=np.float32)

    # Compute similarity transform
    from cv2 import estimateAffinePartial2D, warpAffine
    M, _ = estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
    aligned_face = warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=0)
    return aligned_face

def extract_arcface_embedding(image_path):
    """
    Use InsightFace FaceAnalysis to detect and extract embedding from an image file.
    """
    if arcface_app is None:
        logger.error("FaceAnalysis not initialized.")
        return None, 'FaceAnalysis not initialized'
    img = cv2.imread(image_path)
    if img is None:
        logger.info(f"Could not read image: {image_path}")
        return None, 'Could not read image'
    faces = arcface_app.get(img)
    if not faces:
        logger.info(f"No face detected in image: {image_path}")
        return None, 'No face detected'
    embedding = faces[0].embedding
    if embedding is None or np.isnan(embedding).any() or np.isinf(embedding).any():
        logger.info("Invalid embedding values.")
        return None, 'Invalid embedding values'
    norm = np.linalg.norm(embedding)
    if norm == 0:
        logger.info("Zero norm embedding.")
        return None, 'Zero norm embedding'
    embedding = embedding / norm
    logger.info(f"Valid embedding generated, shape: {embedding.shape}")
    return embedding, None

@app.route('/')
def home():
    return redirect('/live_detection_page')

@app.route('/add_employee', methods=['POST'])
def add_employee():
    name = request.form.get('name')
    employee_id = request.form.get('employee_id')
    description = request.form.get('description')
    files = request.files.getlist('images')
    if not name or not employee_id or not files:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    embeddings = []
    image_filenames = []
    debug_msgs = []

    try:
        employee = Employee(
            name=name,
            employee_id=employee_id,
            description=description
        )
        db.session.add(employee)
        db.session.flush()  # Get employee.id before commit

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                embedding, debug_message = extract_arcface_embedding(filepath)
                debug_msgs.append(f"{filename}: {debug_message}")
                if embedding is not None:
                    embeddings.append(embedding)
                    image_filenames.append(filename)
                    emp_img = EmployeeImage(employee_id=employee.id, image_filename=filename, arcface_embedding=embedding)
                    db.session.add(emp_img)
                else:
                    print(f"[DEBUG] Could not process {filename}: {debug_message}")

        if not embeddings:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': 'Failed to detect a face in any of the uploaded images.', 'debug': debug_msgs}), 400

        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Employee added successfully', 'debug': debug_msgs})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e), 'debug': debug_msgs}), 500

@app.route('/employees', methods=['GET'])
def list_employees():
    employees = Employee.query.all()
    result = []
    for e in employees:
        images = [url_for('static', filename=f'uploads/{img.image_filename}', _external=True) for img in e.images]
        image_url = images[0] if images else url_for('static', filename='default.jpg', _external=True)
        result.append({
            'id': e.id,
            'name': e.name,
            'employee_id': e.employee_id,
            'description': e.description,
            'image_url': image_url,
            'images': images
        })
    return jsonify(result)

@app.route('/edit_employee/<int:employee_id>', methods=['POST'])
def edit_employee(employee_id):
    try:
        employee = Employee.query.get_or_404(employee_id)
        employee.name = request.form['name']
        employee.employee_id = request.form['employee_id']
        employee.description = request.form['description']
        images = request.files.getlist('images')
        new_embeddings = []
        debug_msgs = []
        if images and images[0].filename:
            for image in images:
                if image and allowed_file(image.filename):
                    filename = secure_filename(image.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image.save(filepath)
                    embedding, debug_message = extract_arcface_embedding(filepath)
                    debug_msgs.append(f"{filename}: {debug_message}")
                    if embedding is not None:
                        new_embeddings.append(embedding)
                        emp_img = EmployeeImage(employee_id=employee.id, image_filename=filename, arcface_embedding=embedding)
                        db.session.add(emp_img)
                    else:
                        print(f"[DEBUG] Could not process {filename}: {debug_message}")
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Employee updated successfully', 'debug': debug_msgs})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_employee/<int:employee_id>', methods=['POST'])
def delete_employee(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], img.image_filename) for img in employee.images]
    try:
        db.session.delete(employee)
        db.session.commit()
        for image_path in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)
        return jsonify({'status': 'success', 'message': 'Employee deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500    

# def cosine_similarity(a, b):
#     from numpy import dot
#     from numpy.linalg import norm
#     if a is None or b is None:
#         return 0.0
#     return float(dot(a, b) / (norm(a) * norm(b)))

def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    print(f"[DEBUG] Cosine similarity inputs: a={a}, b={b}")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Add this helper function near the top of the file (after cosine_similarity or in a utils section)
def recognize_employee(embedding, known_embeddings, known_names, threshold=0.5, top_n=3, min_margin=0.07):
    print(f"[DEBUG] Recognizing employee with embedding: {embedding}, known_embeddings: {len(known_embeddings)}, known_names: {len(known_names)}")
    """.
    Recognize an employee using top-N voting and improved thresholding.
    Returns (name, confidence, debug_info)
    """
    if embedding is None or not known_embeddings:
        return "Unknown", 0.0, {"reason": "No embedding or no known embeddings"}
    sims = [cosine_similarity(embedding, kemb) for kemb in known_embeddings]
    if not sims:
        return "Unknown", 0.0, {"reason": "No similarity scores"}
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    top_scores = [sims[i] for i in top_indices]
    top_names = [known_names[i] for i in top_indices]
    debug_info = {"all_scores": sims, "top_scores": top_scores, "top_names": top_names}
    # Main decision: top-1 must be above threshold and margin over top-2
    if top_scores[0] > threshold:
        if top_n > 1 and (top_scores[0] - top_scores[1]) < min_margin:
            debug_info["reason"] = f"Top-1 margin too small: {top_scores[0] - top_scores[1]:.3f}"
            return "Unknown", top_scores[0], debug_info
        debug_info["reason"] = "Recognized"
        return top_names[0], top_scores[0], debug_info
    debug_info["reason"] = f"Top-1 below threshold: {top_scores[0]:.3f} < {threshold}"
    return "Unknown", top_scores[0], debug_info

# --- Real-time Camera Processing --- #

camera_threads = {}
camera_locks = {}
stop_flags = {}
face_trackers = {}  # Dictionary to hold a tracker for each camera

# Detection and recognition configuration
DETECTION_CONFIDENCE_THRESHOLD = 0.85  # Increased threshold for higher confidence
MIN_FACE_SIZE = 60  # Increased minimum face size for better quality
MIN_ASPECT_RATIO = 0.8  # Tightened aspect ratio range
MAX_ASPECT_RATIO = 1.2
MIN_BLURRINESS = 50  # Increased minimum sharpness
RECOGNITION_THRESHOLD = 0.6  # Minimum similarity score for recognition
MIN_RECOGNITION_MARGIN = 0.1  # Minimum difference between top 2 matches
MAX_YAW_ANGLE = 30  # Maximum face rotation angle in degrees
MAX_DETECTION_FAILURES = 3  # Maximum consecutive detection failures before resetting

# Update gen_frames to use MTCNN for detection and ArcFace for recognition only
def gen_frames(video_filename=None, camera_feed_id=None):
    from flask import current_app
    with app.app_context():
        print("[DEBUG] gen_frames called with video_filename=", video_filename, "camera_feed_id=", camera_feed_id)
        # Get all required data before starting the generator
        try:
            employees = Employee.query.all()
            known_arcface_embeddings = []
            known_names = []
            for e in employees:
                img_embeddings = [img.arcface_embedding for img in e.images if img.arcface_embedding is not None]
                if img_embeddings:
                    avg_embedding = np.mean(img_embeddings, axis=0)
                    known_arcface_embeddings.append(avg_embedding)
                    known_names.append(e.name)
                    print(f"[DEBUG] Employee {e.name} has {len(img_embeddings)} embeddings.")
                else:
                    print(f"[DEBUG] No valid embeddings for employee {e.name}")
            print(f"[DEBUG] Total employees: {len(employees)}, with embeddings: {len(known_arcface_embeddings)}")
        except Exception as e:
            print(f"[DEBUG] Exception during employee DB query: {e}")
            error_frame = generate_error_frame(f"DB error: {str(e)}")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            print("[DEBUG] Yielding DB error frame")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return

        camera = None
        frame_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        try:
            if camera_feed_id:
                print(f"[DEBUG] Looking up CameraFeed with id {camera_feed_id}")
                camera_feed = CameraFeed.query.get(camera_feed_id)
                if not camera_feed or not camera_feed.is_active:
                    print(f"[DEBUG] CameraFeed not found or not active: {camera_feed}")
                    return
                if camera_feed.camera_type == 'device':
                    try:
                        cam_index = int(camera_feed.camera_url)
                        print(f"[DEBUG] Attempting to open device camera index {cam_index}")
                        test_cam = cv2.VideoCapture(cam_index)
                        if not test_cam.isOpened():
                            print(f"[DEBUG] Camera device index {cam_index} not available")
                            error_frame = generate_error_frame(f"Camera device index {cam_index} not available")
                            ret, buffer = cv2.imencode('.jpg', error_frame)
                            frame_bytes = buffer.tobytes()
                            print("[DEBUG] Yielding device not available error frame")
                            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            test_cam.release()
                            return
                        test_cam.release()
                        camera = cv2.VideoCapture(cam_index)
                        print(f"[DEBUG] Device camera {cam_index} opened: {camera.isOpened()}")
                    except ValueError:
                        print(f"[DEBUG] Invalid camera index, trying to find available camera")
                        camera = find_available_camera()
                else:
                    rtsp_url = camera_feed.camera_url
                    rtsp_url = encode_rtsp_url(rtsp_url)
                    if "rtsp://" in rtsp_url and "rtsp_transport" not in rtsp_url:
                        sep = '&' if '?' in rtsp_url else '?'
                        rtsp_url += f"{sep}rtsp_transport=tcp"
                    print(f"[DEBUG] Attempting to open RTSP stream: {rtsp_url}")
                    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    retry_count = 0
                    max_retries = 5
                    while (camera is None or not camera.isOpened()) and retry_count < max_retries:
                        print(f"[WARN] Failed to open RTSP stream. Retrying {retry_count+1}/{max_retries}...")
                        time.sleep(2)
                        camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        retry_count += 1
                    print(f"[DEBUG] RTSP camera opened: {camera.isOpened()}")
                    if camera is None or not camera.isOpened():
                        print(f"[DEBUG] Camera not available (RTSP connect failed)")
                        error_frame = generate_error_frame("Camera not available (RTSP connect failed)")
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        frame_bytes = buffer.tobytes()
                        print("[DEBUG] Yielding RTSP not available error frame")
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        return
            elif video_filename:
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                print(f"[DEBUG] Attempting to open video file: {video_path}")
                camera = cv2.VideoCapture(video_path)
                print(f"[DEBUG] Video file camera opened: {camera.isOpened()}")
            else:
                print(f"[DEBUG] No camera_feed_id or video_filename, trying to find available camera")
                camera = find_available_camera()
                print(f"[DEBUG] Available camera opened: {camera.isOpened() if camera else None}")

            if camera is None or not camera.isOpened():
                print(f"[DEBUG] Camera not available at all")
                error_frame = generate_error_frame("Camera not available")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                print("[DEBUG] Yielding camera not available error frame")
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return

            def camera_worker():
                print("[DEBUG] camera_worker started")
                frame_count = 0
                start_time = time.time()
                fps = 0.0
                process_every_n = 5  # Increased for speed
                last_boxes = []
                last_names = []
                last_confidences = []

                if not camera.isOpened():
                    print("[ERROR] Camera stream could not be opened. Check RTSP URL, credentials, and permissions.")
                    error_frame = generate_error_frame("Camera not available (RTSP connect failed)")
                    try:
                        frame_queue.put(error_frame)
                    except Exception:
                        pass
                    return

                while not stop_event.is_set():
                    try:
                        success, frame = camera.read()
                        # print(f"[DEBUG] camera.read() success={success} frame={'not None' if frame is not None else 'None'}")
                        if not success or frame is None:
                            time.sleep(0.01)
                            continue
                        frame_count += 1
                        process_this_frame = (frame_count % process_every_n == 0)
                        if process_this_frame:
                            # print(f"[DEBUG] Processing frame {frame_count}")
                            # Resize for faster detection
                            scale = 0.5
                            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                            faces = arcface_app.get(frame)
                            boxes = []
                            names = []
                            confidences = []
                            for face in faces:
                                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                                embedding = face.embedding
                                name = "Unknown"
                                confidence = 0.0
                                if embedding is not None:
                                    name, confidence, debug_info = recognize_employee(
                                        embedding, known_arcface_embeddings, known_names, threshold=0.5, top_n=3, min_margin=0.07)
                                boxes.append((x1, y1, x2, y2))
                                names.append(name)
                                confidences.append(confidence)
                            last_boxes = boxes
                            last_names = names
                            last_confidences = confidences
                        # Draw boxes and labels
                        for (x1, y1, x2, y2), name, confidence in zip(last_boxes, last_names, last_confidences):
                            color = (255, 0, 0) if name != "Unknown" else (0, 255, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        # FPS calculation
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            fps = frame_count / elapsed_time
                        if not frame_queue.empty():
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        # print(f"[DEBUG] Putting frame {frame_count} in queue")
                        frame_queue.put(frame)
                    except Exception as e:
                        print(f"[ERROR] Exception in camera_worker: {e}")
                        error_frame = generate_error_frame(f"Worker error: {str(e)}")
                        try:
                            frame_queue.put(error_frame)
                        except Exception:
                            pass
                        break

            worker_thread = threading.Thread(target=camera_worker, daemon=True)
            worker_thread.start()

            while True:
                try:
                    frame = frame_queue.get(timeout=2)
                    # print(f"[DEBUG] Got frame from queue, encoding and yielding")
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except queue.Empty:
                    print(f"[DEBUG] Frame queue empty, yielding waiting frame")
                    error_frame = generate_error_frame("Waiting for camera...")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"[ERROR] Exception in gen_frames yield loop: {e}")
                    error_frame = generate_error_frame(f"Yield error: {str(e)}")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    break
        except Exception as e:
            print(f"[ERROR] Error in gen_frames: {e}")
            error_frame = generate_error_frame(f"Error: {str(e)}")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            print("[DEBUG] Yielding top-level error frame")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            stop_event.set()
            if camera is not None:
                camera.release()
            print("[DEBUG] gen_frames finished, camera released")

def find_available_camera():
    """Find an available camera by trying different device indices"""
    for i in range(4):  # Try cameras 0-3
        try:
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                # Test if we can actually read a frame
                ret, frame = camera.read()
                if ret and frame is not None:
                    return camera
                else:
                    camera.release()
            else:
                camera.release()
        except Exception as e:
            print(f"Error trying camera {i}: {e}")
            continue
    return None

def generate_error_frame(message):
    """Generate an error frame with a message"""
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add error text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)

    # Calculate position to center the text
    x = (frame.shape[1] - text_width) // 2
    y = (frame.shape[0] + text_height) // 2

    # Add text
    cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)

    # Add additional help text
    help_text = "Check camera connection or try video upload"
    (help_width, help_height), _ = cv2.getTextSize(help_text, font, 0.7, 1)
    help_x = (frame.shape[1] - help_width) // 2
    help_y = y + 50
    cv2.putText(frame, help_text, (help_x, help_y), font, 0.7, (200, 200, 200), 1)

    return frame 

# --- Web Pages --- #
@app.route('/live_detection')
def live_detection():
    video_filename = request.args.get('video')
    camera_feed_id = request.args.get('camera_feed_id')
    if camera_feed_id:
        camera_feed_id = int(camera_feed_id)
    return app.response_class(gen_frames(video_filename, camera_feed_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_detection_page', methods=['GET'])
def live_detection_page():
    cameras = CameraFeed.query.filter_by(is_active=True).all()
    return render_template('live_detection.html', cameras=cameras)

@app.route('/add_employee_form', methods=['GET'])
def add_employee_form():
    return render_template('add_employee.html')

@app.route('/employees_page', methods=['GET'])
def employees_page():
    return render_template('employees.html')

@app.route('/attendance_logs', methods=['GET'])
def attendance_logs():
    logs = AttendanceLog.query.order_by(AttendanceLog.timestamp.desc()).all()
    # Get all employees for image lookup, using trimmed, lowercased names as keys
    employees = {e.name.strip().lower(): e for e in Employee.query.all()}
    default_image = 'default.jpg'
    return jsonify([
        {
            'id': log.id,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'employee_name': log.employee_name,
            'attendance_type': log.attendance_type,
            'camera_feed_name': log.camera_feed_name,
            # Refactored image_url assignment for clarity and efficiency
            'image_url': (lambda employee: url_for('static', filename=f'uploads/{employee.image_filename if employee else default_image}', _external=True))(
                employees.get((log.employee_name or '').strip().lower())
            )
        } for log in logs
    ])

@app.route('/delete_attendance_log/<int:log_id>', methods=['DELETE'])
def delete_attendance_log(log_id):
    try:
        log = AttendanceLog.query.get_or_404(log_id)
        db.session.delete(log)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Attendance log deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_all_attendance_logs', methods=['DELETE'])
def delete_all_attendance_logs():
    try:
        AttendanceLog.query.delete()
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'All attendance logs deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/attendance_logs_page', methods=['GET'])
def attendance_logs_page():
    return render_template('attendance_logs.html')

# --- API Endpoints for Camera/Employee Management (remains the same) --- #
# Camera Feed Management Routes
@app.route('/add_camera_feed', methods=['POST'])
def add_camera_feed():
    name = request.form.get('name')
    camera_url = request.form.get('camera_url')
    camera_type = request.form.get('camera_type')
    location = request.form.get('location')
    description = request.form.get('description')

    if not name or not camera_url or not camera_type:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    try:
        camera_feed = CameraFeed(
            name=name,
            camera_url=camera_url,
            camera_type=camera_type,
            location=location,
            description=description
        )
        db.session.add(camera_feed)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Camera feed added successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add camera feed
@app.route('/camera_feeds', methods=['GET'])
def list_camera_feeds():
    camera_feeds = CameraFeed.query.all()
    result = []
    for cf in camera_feeds:
        result.append({
            'id': cf.id,
            'name': cf.name,
            'camera_url': cf.camera_url,
            'camera_type': cf.camera_type,
            'location': cf.location,
            'description': cf.description,
            'is_active': cf.is_active,
            'created_at': cf.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(result)

@app.route('/edit_camera_feed/<int:camera_feed_id>', methods=['POST'])
def edit_camera_feed(camera_feed_id):
    camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
    name = request.form.get('name')
    camera_url = request.form.get('camera_url')
    camera_type = request.form.get('camera_type')
    location = request.form.get('location')
    description = request.form.get('description')
    is_active = request.form.get('is_active')

    if name:
        camera_feed.name = name
    if camera_url:
        camera_feed.camera_url = camera_url
    if camera_type:
        camera_feed.camera_type = camera_type
    if location is not None:
        camera_feed.location = location
    if description is not None:
        camera_feed.description = description
    if is_active is not None:
        camera_feed.is_active = is_active.lower() == 'true'

    try:
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Camera feed updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_camera_feed/<int:camera_feed_id>', methods=['POST'])
def delete_camera_feed(camera_feed_id):
    camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
    try:
        db.session.delete(camera_feed)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Camera feed deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/test_camera_feed/<int:camera_feed_id>', methods=['GET'])
def test_camera_feed(camera_feed_id):
    camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
    camera = None
    try:
        if camera_feed.camera_type == 'device':
            try:
                cam_index = int(camera_feed.camera_url)
                camera = cv2.VideoCapture(cam_index)
                if not camera.isOpened():
                    return jsonify({'status': 'error', 'message': f'Camera device index {cam_index} not available'}), 400
            except ValueError:
                camera = find_available_camera()
                if camera is None:
                    return jsonify({'status': 'error', 'message': 'No camera devices available'}), 400
        else:
            rtsp_url = encode_rtsp_url(camera_feed.camera_url)
            camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Cannot connect to camera feed'}), 400

        ret, frame = camera.read()
        if ret and frame is not None:
            return jsonify({'status': 'success', 'message': 'Camera feed is working'})
        else:
            return jsonify({'status': 'error', 'message': 'Camera feed is not working - no video signal'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error testing camera: {str(e)}'}), 500
    finally:
        if camera is not None:
            camera.release()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5000)
def test_camera_feed(camera_feed_id):
    camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
    camera = None
    try:
        if camera_feed.camera_type == 'device':
            try:
                cam_index = int(camera_feed.camera_url)
                # Check if the device index is available
                test_cam = cv2.VideoCapture(cam_index)
                if not test_cam.isOpened():
                    return jsonify({'status': 'error', 'message': f'Camera device index {cam_index} not available'}), 400
                test_cam.release()
                camera = cv2.VideoCapture(cam_index)
            except ValueError:
                # Try to find an available camera
                camera = find_available_camera()
                if camera is None:
                    return jsonify({'status': 'error', 'message': 'No camera devices available'}), 400
        else:
            camera = cv2.VideoCapture(camera_feed.camera_url, cv2.CAP_FFMPEG)

        if camera is None or not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Cannot connect to camera feed'}), 400

        ret, frame = camera.read()
        if ret and frame is not None:
            return jsonify({'status': 'success', 'message': 'Camera feed is working'})
        else:
            return jsonify({'status': 'error', 'message': 'Camera feed is not working - no video signal'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error testing camera: {str(e)}'}), 500
    finally:
        if camera is not None:
            camera.release()

@app.route('/add_camera_feed_form', methods=['GET'])
def add_camera_feed_form():
    return render_template('add_camera_feed.html')

@app.route('/camera_feeds_page')
def camera_feeds_page():
    feeds = CameraFeed.query.all()
    return render_template('camera_feeds.html', feeds=feeds)

@app.route('/cctv_detection_page', methods=['GET'])
def cctv_detection_page():
    return render_template('cctv_detection.html')

@app.route('/detection_config', methods=['GET'])
def get_detection_config():
    return jsonify({
        'cooldown_seconds': app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
    })

@app.route('/employee_status_page', methods=['GET'])
def employee_status_page():
    return render_template('employee_status.html')

@app.route('/employee_status_data', methods=['GET'])
def get_employee_status_data():
    """Get real-time employee status data"""
    try:
        # Get all employees
        employees = Employee.query.all()
        employee_data = []

        # Get recent attendance logs
        cooldown_seconds = app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
        recent_time = datetime.utcnow() - timedelta(seconds=cooldown_seconds)

        recent_attendances = AttendanceLog.query.filter(
            AttendanceLog.timestamp >= recent_time
        ).all()

        # Create a map of recent detections
        recent_attendance_map = {}
        for attendance in recent_attendances:
            if attendance.employee_name not in recent_attendance_map:
                recent_attendance_map[attendance.employee_name] = []
            recent_attendance_map[attendance.employee_name].append({
                'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'attendance_type': attendance.attendance_type,
                'camera_feed_name': attendance.camera_feed_name
            })

        # Build status data for each employee
        for employee in employees:
            recent_attendances_for_employee = recent_attendance_map.get(employee.name, [])
            is_present = len(recent_attendances_for_employee) > 0

            employee_data.append({
                'id': employee.id,
                'name': employee.name,
                'employee_id': employee.employee_id,
                'image_url': url_for('static', filename=f'uploads/{employee.image_filename}', _external=True),
                'status': 'present' if is_present else 'absent',
                'attendance_count': len(recent_attendances_for_employee),
                'last_detected': recent_attendances_for_employee[-1]['timestamp'] if recent_attendances_for_employee else None,
                'recent_attendances': recent_attendances_for_employee
            })

        return jsonify({
            'employees': employee_data,
            'summary': {
                'total': len(employees),
                'present': len([e for e in employee_data if e['status'] == 'present']),
                'absent': len([e for e in employee_data if e['status'] == 'absent']),
                'attendance_rate': round((len([e for e in employee_data if e['status'] == 'present']) / len(employees)) * 100) if employees else 0,
                'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, use_reloader=False)    