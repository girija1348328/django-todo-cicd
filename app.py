# import eventlet
# eventlet.monkey_patch()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import face_recognition
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import time
from datetime import datetime, timedelta
from threading import Thread
import threading
import queue
import dlib
import urllib.parse
from mtcnn import MTCNN
from insightface.app import FaceAnalysis
from flask_migrate import Migrate
import onnxruntime as ort
from insightface.model_zoo import ArcFaceONNX
from collections import deque
import base64
from face_tracker import FaceTracker

print("OpenCV version:", cv2.__version__)
build_info = cv2.getBuildInformation()
ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]
print("FFMPEG in build info:", ffmpeg_support)

# FFMPEG backend will be used explicitly when opening cameras
print("FFMPEG backend will be used for RTSP streams")

print(ort.get_device())

session = ort.InferenceSession("insightface_repo/model_zoo/model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(session.get_providers())


model = ArcFaceONNX("insightface_repo/model_zoo/model.onnx")
model.prepare(ctx_id=0)
mtcnn_detector = MTCNN()

# mtcnn = MTCNN(keep_all=True, device='cuda')

if getattr(dlib, 'DLIB_USE_CUDA', False):
    print("Running on GPU")
else:
    print("Running on CPU")


app = Flask(__name__)

# Configurations
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['DETECTION_COOLDOWN_SECONDS'] = 30  # Time window to prevent duplicate detections


# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SocketIO
from flask_socketio import SocketIO
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    image_filename = db.Column(db.String(200), nullable=False)
    arcface_embedding = db.Column(db.PickleType, nullable=True)  # Store numpy array as binary

    def __repr__(self):
        return f'<Employee {self.name}>'

class CameraFeed(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    camera_url = db.Column(db.String(500), nullable=False)  # RTSP URL, IP camera URL, or device index
    camera_type = db.Column(db.String(50), nullable=False)  # 'rtsp', 'ip', 'device', 'cctv'
    location = db.Column(db.String(200), nullable=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<CameraFeed {self.name}>'

class AttendanceLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    employee_name = db.Column(db.String(100), nullable=False)
    attendance_type = db.Column(db.String(20), nullable=False)  # 'video', 'live', 'cctv'
    camera_feed_id = db.Column(db.Integer, db.ForeignKey('camera_feed.id'), nullable=True)
    camera_feed_name = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<AttendanceLog {self.employee_name} at {self.timestamp}>'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def check_ffmpeg_support():
#     import cv2
#     build_info = cv2.getBuildInformation()
#     # Check for FFMPEG support (can be "FFMPEG: YES" or "FFMPEG: YES (prebuilt binaries)")
#     # The actual format is "      FFMPEG:                      YES (prebuilt binaries)"
#     if 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]:
#         print("[INFO] OpenCV FFMPEG support detected.")
#         return True
#     else:
#         print("[ERROR] OpenCV is not built with FFMPEG support. RTSP streams will not work.")
#         print("[SOLUTION] Install opencv-python-headless instead of opencv-python")
#         print("[SOLUTION] Run: pip uninstall opencv-python && pip install opencv-python-headless")
#         return False

# def check_system_ffmpeg():
#     """Check if FFMPEG is available on the system"""
#     import subprocess
#     try:
#         result = subprocess.run(['ffmpeg', '-version'], 
#                               capture_output=True, text=True, timeout=5)
#         if result.returncode == 0:
#             print("[INFO] System FFMPEG detected")
#             return True
#     except (subprocess.TimeoutExpired, FileNotFoundError):
#         pass
    
#     # Check for local FFMPEG installation
#     import os
#     local_ffmpeg = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
#     if os.path.exists(local_ffmpeg):
#         try:
#             result = subprocess.run([local_ffmpeg, '-version'], 
#                                   capture_output=True, text=True, timeout=5)
#             if result.returncode == 0:
#                 print("[INFO] Local FFMPEG detected")
#                 # Add to PATH for current session
#                 ffmpeg_bin = os.path.join(os.getcwd(), "ffmpeg", "bin")
#                 if ffmpeg_bin not in os.environ.get('PATH', ''):
#                     os.environ['PATH'] = f"{ffmpeg_bin};{os.environ.get('PATH', '')}"
#                 return True
#         except subprocess.TimeoutExpired:
#             pass
    
#     print("[WARNING] FFMPEG not found on system. RTSP streams may not work properly.")
#     print("[SOLUTION] Install FFMPEG or run the install_ffmpeg_windows.py script")
#     return False

# # Check both OpenCV FFMPEG support and system FFMPEG
# opencv_ffmpeg_ok = check_ffmpeg_support()
# system_ffmpeg_ok = check_system_ffmpeg()

# if not opencv_ffmpeg_ok:
#     print("[CRITICAL] OpenCV FFMPEG support is required for RTSP streams!")
#     print("[ACTION] Please fix this before using RTSP camera feeds.")


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

# # Add a new column to Employee for ArcFace embedding (if not already present)
# # If using Alembic or migrations, this should be handled there. For now, add in-memory only for demonstration.
# if not hasattr(Employee, 'arcface_embedding'):
#     from sqlalchemy import PickleType
#     Employee.arcface_embedding = db.Column(PickleType, nullable=True)

# Initialize ArcFace model globally
ctx_id = -1
arcface_app = FaceAnalysis(name='buffalo_l')
arcface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

def extract_arcface_embedding_from_crop(face_crop):
    if face_crop.shape[2] == 3:
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    else:
        rgb_crop = face_crop
    faces = arcface_app.get(rgb_crop)
    if faces:
        return faces[0].embedding
    return None

def align_face_by_keypoints(img, keypoints, output_size=(112, 112)):
    ref_pts = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src_pts = np.array([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']], dtype=np.float32)
    from cv2 import estimateAffinePartial2D, warpAffine
    M, _ = estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
    aligned_face = warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=0)
    return aligned_face

def extract_arcface_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not read image"
    detections = mtcnn_detector.detect_faces(img)
    if not detections:
        return None, "No face detected by MTCNN"
    first_detection = detections[0]
    keypoints = first_detection['keypoints']
    aligned_face = align_face_by_keypoints(img, keypoints, output_size=(112, 112))
    embedding = extract_arcface_embedding_from_crop(aligned_face)
    if embedding is not None:
        return embedding, 'Embedding extracted successfully.'
    return None, 'ArcFace failed on the aligned crop.'

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
        return jsonify({'status': 'error', 'message': 'Missing or invalid data'}), 400

    embeddings = []
    saved_filename = None

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            embedding, debug_message = extract_arcface_embedding(filepath)
            if embedding is not None:
                embeddings.append(embedding)
                if saved_filename is None:
                    saved_filename = filename # Save the first valid image's name
            else:
                print(f"Could not process {filename}: {debug_message}")

    if not embeddings:
        return jsonify({'status': 'error', 'message': 'Failed to detect a face in any of the uploaded images.'}), 400

    # Calculate the average embedding
    arcface_embedding = np.mean(embeddings, axis=0)

    try:
        employee = Employee(
            name=name,
            employee_id=employee_id,
            description=description,
            image_filename=saved_filename,
            arcface_embedding=arcface_embedding
        )
        db.session.add(employee)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Employee added successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/employees', methods=['GET'])
def list_employees():
    employees = Employee.query.all()
    result = []
    for e in employees:
        result.append({
            'id': e.id,
            'name': e.name,
            'employee_id': e.employee_id,
            'description': e.description,
            'image_url': url_for('static', filename=f'uploads/{e.image_filename}', _external=True)
        })
    return jsonify(result)

@app.route('/edit_employee/<int:employee_id>', methods=['POST'])
def edit_employee(employee_id):
    try:
        employee = Employee.query.get_or_404(employee_id)

        # Update textual details
        employee.name = request.form['name']
        employee.employee_id = request.form['employee_id']
        employee.description = request.form['description']

        images = request.files.getlist('images')
        new_embeddings = []

        # Process new images if any are uploaded
        if images and images[0].filename:
            for image in images:
                if image and allowed_file(image.filename):
                    # We don't need to save the new files, just extract embeddings
                    # To save memory, process from stream
                    in_memory_file = io.BytesIO()
                    image.save(in_memory_file)
                    in_memory_file.seek(0)
                    img_array = np.frombuffer(in_memory_file.read(), np.uint8)
                    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if img_cv is None: continue

                    dets = mtcnn_detector.detect_faces(img_cv)
                    if not dets: continue

                    det = dets[0] # Assume first face
                    embedding = None
                    try:
                        aligned_face = align_face_by_keypoints(img_cv, det['keypoints'])
                        embedding = extract_arcface_embedding_from_crop(aligned_face)
                    except Exception:
                        x, y, w, h = det['box']
                        face_crop = img_cv[y:y+h, x:x+w]
                        if face_crop.size > 0:
                            embedding = extract_arcface_embedding_from_crop(face_crop)
                    
                    if embedding is not None:
                        new_embeddings.append(embedding)

            if new_embeddings:
                all_embeddings = []
                if employee.arcface_embedding is not None:
                    # Convert existing embedding from bytes back to numpy array
                    existing_embedding = np.frombuffer(employee.arcface_embedding, dtype=np.float32)
                    all_embeddings.append(existing_embedding)
                
                all_embeddings.extend(new_embeddings)
                
                # Calculate new average and update
                avg_embedding = np.mean(all_embeddings, axis=0)
                employee.arcface_embedding = avg_embedding.tobytes()

        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Employee updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/delete_employee/<int:employee_id>', methods=['POST'])
def delete_employee(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], employee.image_filename)
    try:
        db.session.delete(employee)
        db.session.commit()
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'status': 'success', 'message': 'Employee deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500    

def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Real-time Camera Processing --- #

camera_threads = {}
camera_locks = {}
stop_flags = {}
face_trackers = {}  # Dictionary to hold a tracker for each camera

# Update gen_frames to use MTCNN for detection and ArcFace for recognition only
def gen_frames(video_filename=None, camera_feed_id=None):
    ctx = app.app_context()
    ctx.push()
    camera = None
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    try:
        employees = Employee.query.all()
        known_arcface_embeddings = [getattr(e, 'arcface_embedding', None) for e in employees]
        known_names = [e.name for e in employees]

        if camera_feed_id:
            camera_feed = CameraFeed.query.get(camera_feed_id)
            if not camera_feed or not camera_feed.is_active:
                return
            if camera_feed.camera_type == 'device':
                try:
                    cam_index = int(camera_feed.camera_url)
                    test_cam = cv2.VideoCapture(cam_index)
                    if not test_cam.isOpened():
                        error_frame = generate_error_frame(f"Camera device index {cam_index} not available")
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        test_cam.release()
                        return
                    test_cam.release()
                    camera = cv2.VideoCapture(cam_index)
                except ValueError:
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
                if camera is None or not camera.isOpened():
                    error_frame = generate_error_frame("Camera not available (RTSP connect failed)")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    return
        elif video_filename:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            camera = cv2.VideoCapture(video_path)
        else:
            camera = find_available_camera()

        if camera is None or not camera.isOpened():
            error_frame = generate_error_frame("Camera not available")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
        
        def camera_worker(camera_id, stop_event):
        with app.app_context():
        camera_feed = CameraFeed.query.get(camera_id)
        if camera_feed.camera_type == 'device':
            cap = cv2.VideoCapture(int(camera_feed.camera_url))
        else:
            cap = cv2.VideoCapture(camera_feed.camera_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_id}")
            return
        employees = Employee.query.filter(Employee.arcface_embedding.isnot(None)).all()
        known_arcface_embeddings = [emp.arcface_embedding for emp in employees]
        known_names = [emp.name for emp in employees]
        tracker = face_trackers[camera_id]
        frame_skip = 2
        frame_count = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue
            frame_count += 1
            if frame_count % frame_skip != 0:
                # Draw previous stable boxes on skipped frames for smoother visuals
                stable_objects = tracker.get_stable_names()
                for obj_id, (box, name) in stable_objects.items():
                    x, y, x2, y2 = box
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    label = f"{name}"
                    cv2.rectangle(frame, (x, y - 20), (x + len(label)*9, y), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                socketio.emit(f'video_frame_{camera_id}', {'image': base64.b64encode(buffer).decode('utf-8')})
                continue
            
            height, width, _ = frame.shape
            scale = 320 / width
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            detections = mtcnn_detector.detect_faces(rgb_small_frame)
            boxes = []
            names = []
            for det in detections:
                if det['confidence'] < 0.90:
                    continue
                x, y, w, h = [int(v / scale) for v in det['box']]
                x2, y2 = x + w, y + h
                embedding = None
                try:
                    aligned_face = align_face_by_keypoints(rgb_small_frame, det['keypoints'])
                    embedding = extract_arcface_embedding_from_crop(aligned_face)
                except Exception:
                    face_crop = frame[y:y2, x:x2]
                    if face_crop.size > 0:
                        embedding = extract_arcface_embedding_from_crop(face_crop)
                name = "Unknown"
                if embedding is not None:
                    sims = [cosine_similarity(embedding, kemb) for kemb in known_arcface_embeddings]
                    if sims and max(sims) > 0.5:
                        name = known_names[sims.index(max(sims))]
                boxes.append((x, y, x2, y2))
                names.append(name)
            
            stable_objects = tracker.update(boxes, names)
            for obj_id, (box, name) in stable_objects.items():
                x, y, x2, y2 = box
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                label = f"{name}"
                cv2.rectangle(frame, (x, y - 20), (x + len(label)*9, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            socketio.emit(f'video_frame_{camera_id}', {'image': base64.b64encode(buffer).decode('utf-8')})
        cap.release()

# # Initialize ArcFace model globally
# import os
# ctx_id = -1
# arcface_app = FaceAnalysis(name='buffalo_l')
# arcface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

# # Helper to extract ArcFace embedding from a face crop
# import cv2
# import numpy as np

# def extract_arcface_embedding_from_crop(face_crop):
#     # ArcFace expects RGB
#     if face_crop.shape[2] == 3:
#         rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
#     else:
#         rgb_crop = face_crop
#     faces = arcface_app.get(rgb_crop)
#     print(f"[DEBUG] ArcFace faces: {faces}")
#     if faces and hasattr(faces[0], 'embedding'):
#         return faces[0].embedding
#     return None

# def align_face_by_keypoints(img, keypoints, output_size=(112, 112)):
#     # Standard ArcFace reference points for 112x112
#     ref_pts = np.array([
#         [38.2946, 51.6963],   # left eye
#         [73.5318, 51.5014],   # right eye
#         [56.0252, 71.7366],   # nose
#         [41.5493, 92.3655],   # left mouth
#         [70.7299, 92.2041]    # right mouth
#     ], dtype=np.float32)

#     src_pts = np.array([
#         keypoints['left_eye'],
#         keypoints['right_eye'],
#         keypoints['nose'],
#         keypoints['mouth_left'],
#         keypoints['mouth_right']
#     ], dtype=np.float32)

#     # Compute similarity transform
#     from cv2 import estimateAffinePartial2D, warpAffine
#     M, _ = estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
#     aligned_face = warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=0)
#     return aligned_face

# def extract_arcface_embedding(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"[DEBUG] Could not read image: {image_path}")
#         return None, 'Could not read image'

#     print(f"[DEBUG] Image shape: {img.shape}")

#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     try:
#         detections = mtcnn_detector.detect_faces(rgb_img)
#     except Exception as e:
#         print(f"[ERROR] MTCNN detection failed: {e}")
#         return None, f'MTCNN detection failed: {e}'

#     print(f"[DEBUG] MTCNN detections: {detections}")
#     if not detections:
#         debug_path = image_path.replace('.jpg', '_debug.jpg').replace('.jpeg', '_debug.jpeg').replace('.png', '_debug.png')
#         cv2.imwrite(debug_path, img)
#         print(f"[DEBUG] No face detected. Saved debug image to: {debug_path}")
#         return None, f'No face detected. Debug image: {debug_path}'

#     det = detections[0]
#     x, y, w, h = det['box']
#     x = max(0, x)
#     y = max(0, y)
#     x2 = min(x + w, img.shape[1])
#     y2 = min(y + h, img.shape[0])
#     face_crop = img[y:y2, x:x2]
#     print(f"[DEBUG] Face crop shape: {face_crop.shape}")

#     keypoints = det['keypoints']
#     crop_keypoints = {k: (v[0] - x, v[1] - y) for k, v in keypoints.items()}

#     # Try alignment first
#     try:
#         aligned_face = align_face_by_keypoints(face_crop, crop_keypoints, output_size=(112, 112))
#         cv2.imwrite("static/uploads/debug_aligned_face.jpg", aligned_face)
#         print("[DEBUG] Aligned face shape:", aligned_face.shape, "dtype:", aligned_face.dtype)
#         embedding = extract_arcface_embedding_from_crop(aligned_face)
#         if embedding is not None:
#             return embedding, None
#     except Exception as e:
#         print(f"[ERROR] Face alignment failed: {e}")

#     # Fallback: crop a square region around the nose and eyes
#     try:
#         nose = crop_keypoints['nose']
#         left_eye = crop_keypoints['left_eye']
#         right_eye = crop_keypoints['right_eye']
#         d_eye = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
#         size = int(max(60, min(face_crop.shape[0], face_crop.shape[1], d_eye * 2)))
#         cx, cy = int(nose[0]), int(nose[1])
#         half = size // 2
#         sx = max(0, cx - half)
#         sy = max(0, cy - half)
#         ex = min(face_crop.shape[1], cx + half)
#         ey = min(face_crop.shape[0], cy + half)
#         square_crop = face_crop[sy:ey, sx:ex]
#         resized_crop = cv2.resize(square_crop, (112, 112))
#         cv2.imwrite("static/uploads/debug_fallback_square_crop.jpg", resized_crop)
#         print("[DEBUG] Fallback square crop shape:", resized_crop.shape, "dtype:", resized_crop.dtype)
#         embedding = extract_arcface_embedding_from_crop(resized_crop)
#         if embedding is not None:
#             return embedding, None
#     except Exception as e:
#         print(f"[ERROR] Fallback square crop failed: {e}")

#     # Final fallback: resize entire crop
#     try:
#         resized_crop = cv2.resize(face_crop, (112, 112))
#         cv2.imwrite("static/uploads/debug_final_fallback_crop.jpg", resized_crop)
#         print("[DEBUG] Final fallback crop shape:", resized_crop.shape, "dtype:", resized_crop.dtype)
#         embedding = extract_arcface_embedding_from_crop(resized_crop)
#         if embedding is not None:
#             return embedding, None
#     except Exception as e:
#         print(f"[ERROR] Final fallback resize failed: {e}")

#     # --- NEW PATCH: Try the full image as a last resort ---
#     try:
#         resized_full = cv2.resize(img, (112, 112))
#         cv2.imwrite("static/uploads/debug_full_image_crop.jpg", resized_full)
#         print("[DEBUG] Full image fallback crop shape:", resized_full.shape, "dtype:", resized_full.dtype)
#         embedding = extract_arcface_embedding_from_crop(resized_full)
#         if embedding is not None:
#             return embedding, None
#     except Exception as e:
#         print(f"[ERROR] Full image fallback failed: {e}")
#     # Save crop for debugging
#     debug_path = image_path.replace('.jpg', '_arcfacefail.jpg').replace('.jpeg', '_arcfacefail.jpeg').replace('.png', '_arcfacefail.png')
#     cv2.imwrite(debug_path, face_crop)
#     print(f"[DEBUG] ArcFace failed. Saved crop to: {debug_path}")
#     return None, f'ArcFace failed. Debug crop: {debug_path}'

# @app.route('/add_employee', methods=['POST'])
# def add_employee():
#     name = request.form.get('name')
#     employee_id = request.form.get('employee_id')
#     description = request.form.get('description')
#     file = request.files.get('image')

#     if not name or not employee_id or file is None or not hasattr(file, 'filename') or not file.filename or not allowed_file(file.filename):
#         return jsonify({'status': 'error', 'message': 'Missing or invalid data'}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     arcface_embedding, debug_message = extract_arcface_embedding(filepath)
#     if arcface_embedding is None:
#         import time
#         for _ in range(5):
#             try:
#                 os.remove(filepath)
#                 break
#             except PermissionError:
#                 time.sleep(0.2)
#         return jsonify({'status': 'error', 'message': f'No face detected or embedding failed. {debug_message}'}), 400

#     try:
#         employee = Employee(
#             name=name,
#             employee_id=employee_id,
#             description=description,
#             image_filename=filename,
#             arcface_embedding=arcface_embedding
#         )
#         db.session.add(employee)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Employee added successfully'})
#     except Exception as e:
#         if os.path.exists(filepath):
#             import time
#             for _ in range(5):
#                 try:
#                     os.remove(filepath)
#                     break
#                 except PermissionError:
#                     time.sleep(0.2)
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/employees', methods=['GET'])
# def list_employees():
#     employees = Employee.query.all()
#     result = []
#     for e in employees:
#         result.append({
#             'id': e.id,
#             'name': e.name,
#             'employee_id': e.employee_id,
#             'description': e.description,
#             'image_url': url_for('static', filename=f'uploads/{e.image_filename}', _external=True)
#         })
#     return jsonify(result)

# @app.route('/edit_employee/<int:employee_id>', methods=['POST'])
# def edit_employee(employee_id):
#     employee = Employee.query.get_or_404(employee_id)
#     name = request.form.get('name')
#     new_employee_id = request.form.get('employee_id')
#     description = request.form.get('description')
#     file = request.files.get('image')

#     if name:
#         employee.name = name
#     if new_employee_id:
#         employee.employee_id = new_employee_id
#     if description is not None:
#         employee.description = description

#     if file and hasattr(file, 'filename') and file.filename and allowed_file(file.filename):
#         old_path = os.path.join(app.config['UPLOAD_FOLDER'], employee.image_filename)
#         if os.path.exists(old_path):
#             os.remove(old_path)
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         arcface_embedding, debug_message = extract_arcface_embedding(filepath)
#         if arcface_embedding is None:
#             os.remove(filepath)
#             return jsonify({'status': 'error', 'message': f'No face detected in new image. {debug_message}'}), 400
#         employee.arcface_embedding = arcface_embedding
#         employee.image_filename = filename

#     try:
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Employee updated successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/delete_employee/<int:employee_id>', methods=['POST'])
# def delete_employee(employee_id):
#     employee = Employee.query.get_or_404(employee_id)
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], employee.image_filename)
#     try:
#         db.session.delete(employee)
#         db.session.commit()
#         if os.path.exists(image_path):
#             os.remove(image_path)
#         return jsonify({'status': 'success', 'message': 'Employee deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     file = request.files.get('video')
#     if file is None or not hasattr(file, 'filename') or not file.filename:
#         return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#     # Do not process video here, just return filename for live detection
#     return jsonify({'status': 'success', 'video_filename': filename})

# def cosine_similarity(a, b):
#     from numpy import dot
#     from numpy.linalg import norm
#     if a is None or b is None:
#         return 0.0
#     return float(dot(a, b) / (norm(a) * norm(b)))

# # Update gen_frames to use MTCNN for detection and ArcFace for recognition only
# def gen_frames(video_filename=None, camera_feed_id=None):
#     ctx = app.app_context()
#     ctx.push()
#     camera = None
#     frame_queue = queue.Queue(maxsize=2)
#     stop_event = threading.Event()
#     try:
#         employees = Employee.query.all()
#         known_arcface_embeddings = [getattr(e, 'arcface_embedding', None) for e in employees]
#         known_names = [e.name for e in employees]

#         if camera_feed_id:
#             camera_feed = CameraFeed.query.get(camera_feed_id)
#             if not camera_feed or not camera_feed.is_active:
#                 return
#             if camera_feed.camera_type == 'device':
#                 try:
#                     cam_index = int(camera_feed.camera_url)
#                     test_cam = cv2.VideoCapture(cam_index)
#                     if not test_cam.isOpened():
#                         error_frame = generate_error_frame(f"Camera device index {cam_index} not available")
#                         ret, buffer = cv2.imencode('.jpg', error_frame)
#                         frame_bytes = buffer.tobytes()
#                         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#                         test_cam.release()
#                         return
#                     test_cam.release()
#                     camera = cv2.VideoCapture(cam_index)
#                 except ValueError:
#                     camera = find_available_camera()
#             else:
#                 rtsp_url = camera_feed.camera_url
#                 rtsp_url = encode_rtsp_url(rtsp_url)
#                 if "rtsp://" in rtsp_url and "rtsp_transport" not in rtsp_url:
#                     sep = '&' if '?' in rtsp_url else '?'
#                     rtsp_url += f"{sep}rtsp_transport=tcp"
#                 print(f"[DEBUG] Attempting to open RTSP stream: {rtsp_url}")
#                 camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#                 retry_count = 0
#                 max_retries = 5
#                 while (camera is None or not camera.isOpened()) and retry_count < max_retries:
#                     print(f"[WARN] Failed to open RTSP stream. Retrying {retry_count+1}/{max_retries}...")
#                     time.sleep(2)
#                     camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#                     retry_count += 1
#                 if camera is None or not camera.isOpened():
#                     error_frame = generate_error_frame("Camera not available (RTSP connect failed)")
#                     ret, buffer = cv2.imencode('.jpg', error_frame)
#                     frame_bytes = buffer.tobytes()
#                     yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#                     return
#         elif video_filename:
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
#             camera = cv2.VideoCapture(video_path)
#         else:
#             camera = find_available_camera()

#         if camera is None or not camera.isOpened():
#             error_frame = generate_error_frame("Camera not available")
#             ret, buffer = cv2.imencode('.jpg', error_frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#             return

#         def camera_worker():
#             with app.app_context():
#                 frame_count = 0
#                 start_time = time.time()
#                 fps = 0.0
#                 process_every_n = 5  # Increased for speed
#                 last_boxes = []
#                 last_names = []
#                 last_confidences = []

#                 if not camera.isOpened():
#                     print("[ERROR] Camera stream could not be opened. Check RTSP URL, credentials, and permissions.")
#                     error_frame = generate_error_frame("Camera not available (RTSP connect failed)")
#                     ret, buffer = cv2.imencode('.jpg', error_frame)
#                     frame_bytes = buffer.tobytes()
#                     frame_queue.put(error_frame)
#                     return

#                 while not stop_event.is_set():
#                     success, frame = camera.read()
#                     if not success or frame is None:
#                         time.sleep(0.01)
#                         continue
#                     frame_count += 1
#                     process_this_frame = (frame_count % process_every_n == 0)
#                     if process_this_frame:
#                         # Resize for faster detection
#                         scale = 0.5
#                         small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
#                         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#                         detections = mtcnn_detector.detect_faces(rgb_small_frame)
#                         height, width = frame.shape[:2]
#                         boxes = []
#                         names = []
#                         confidences = []
#                         for det in detections:
#                             # # Filter by detection confidence
#                             # if 'confidence' in det and det['confidence'] < 0.95:
#                             #     continue  # Skip low-confidence detections
#                             x, y, w, h = det['box']
#                             # Scale coordinates back up
#                             x = int(x / scale)
#                             y = int(y / scale)
#                             w = int(w / scale)
#                             h = int(h / scale)
#                             # Clamp to image boundaries
#                             x = max(0, x)
#                             y = max(0, y)
#                             x2 = min(x + w, width)
#                             y2 = min(y + h, height)
#                             if w < 30 or h < 30:
#                                 continue  # Skip tiny detections
#                             face_crop = frame[y:y2, x:x2]
#                             embedding = extract_arcface_embedding_from_crop(face_crop)
#                             name = "Unknown"
#                             confidence = 0.0
#                             if embedding is not None:
#                                 sims = [cosine_similarity(embedding, kemb) for kemb in known_arcface_embeddings]
#                                 # Increased threshold to 0.5 for stricter recognition
#                                 if sims and max(sims) > 0.4:
#                                     idx = sims.index(max(sims))
#                                     name = known_names[idx]
#                                     confidence = max(sims)
#                             boxes.append((x, y, x2, y2))
#                             names.append(name)
#                             confidences.append(confidence)
#                         last_boxes = boxes
#                         last_names = names
#                         last_confidences = confidences
#                     # Draw boxes and labels
#                     for (x1, y1, x2, y2), name, confidence in zip(last_boxes, last_names, last_confidences):
#                         color = (255, 0, 0) if name != "Unknown" else (0, 255, 255)
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                         label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
#                         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#                     # FPS calculation
#                     elapsed_time = time.time() - start_time
#                     if elapsed_time > 0:
#                         fps = frame_count / elapsed_time
#                     if not frame_queue.empty():
#                         try:
#                             frame_queue.get_nowait()
#                         except queue.Empty:
#                             pass
#                     frame_queue.put(frame)

#         worker_thread = threading.Thread(target=camera_worker, daemon=True)
#         worker_thread.start()

#         while True:
#             try:
#                 frame = frame_queue.get(timeout=2)
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#             except queue.Empty:
#                 error_frame = generate_error_frame("Waiting for camera...")
#                 ret, buffer = cv2.imencode('.jpg', error_frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     except Exception as e:
#         print(f"Error in gen_frames: {e}")
#         error_frame = generate_error_frame(f"Error: {str(e)}")
#         ret, buffer = cv2.imencode('.jpg', error_frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     finally:
#         stop_event.set()
#         if camera is not None:
#             camera.release()
#         ctx.pop()

# def find_available_camera():
#     """Find an available camera by trying different device indices"""
#     for i in range(4):  # Try cameras 0-3
#         try:
#             camera = cv2.VideoCapture(i)
#             if camera.isOpened():
#                 # Test if we can actually read a frame
#                 ret, frame = camera.read()
#                 if ret and frame is not None:
#                     return camera
#                 else:
#                     camera.release()
#             else:
#                 camera.release()
#         except Exception as e:
#             print(f"Error trying camera {i}: {e}")
#             continue
#     return None

# def generate_error_frame(message):
#     """Generate an error frame with a message"""
#     # Create a black frame
#     frame = np.zeros((480, 640, 3), dtype=np.uint8)

#     # Add error text
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     thickness = 2
#     color = (255, 255, 255)

#     # Get text size
#     (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)

#     # Calculate position to center the text
#     x = (frame.shape[1] - text_width) // 2
#     y = (frame.shape[0] + text_height) // 2

#     # Add text
#     cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)

#     # Add additional help text
#     help_text = "Check camera connection or try video upload"
#     (help_width, help_height), _ = cv2.getTextSize(help_text, font, 0.7, 1)
#     help_x = (frame.shape[1] - help_width) // 2
#     help_y = y + 50
#     cv2.putText(frame, help_text, (help_x, help_y), font, 0.7, (200, 200, 200), 1)

#     return frame

# @app.route('/live_detection')
# def live_detection():
#     video_filename = request.args.get('video')
#     camera_feed_id = request.args.get('camera_feed_id')
#     if camera_feed_id:
#         camera_feed_id = int(camera_feed_id)
#     return app.response_class(gen_frames(video_filename, camera_feed_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/add_employee_form', methods=['GET'])
# def add_employee_form():
#     return render_template('add_employee.html')

# @app.route('/employees_page', methods=['GET'])
# def employees_page():
#     return render_template('employees.html')

# @app.route('/live_detection_page', methods=['GET'])
# def live_detection_page():
#     return render_template('live_detection.html')

# @app.route('/upload_video_page', methods=['GET'])
# def upload_video_page():
#     return render_template('upload_video.html')

# @app.route('/attendance_logs', methods=['GET'])
# def attendance_logs():
#     logs = AttendanceLog.query.order_by(AttendanceLog.timestamp.desc()).all()
#     # Get all employees for image lookup, using trimmed, lowercased names as keys
#     employees = {e.name.strip().lower(): e for e in Employee.query.all()}
#     default_image = 'default.jpg'
#     return jsonify([
#         {
#             'id': log.id,
#             'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#             'employee_name': log.employee_name,
#             'attendance_type': log.attendance_type,
#             'camera_feed_name': log.camera_feed_name,
#             # Refactored image_url assignment for clarity and efficiency
#             'image_url': (lambda employee: url_for('static', filename=f'uploads/{employee.image_filename if employee else default_image}', _external=True))(
#                 employees.get((log.employee_name or '').strip().lower())
#             )
#         } for log in logs
#     ])

# @app.route('/delete_attendance_log/<int:log_id>', methods=['DELETE'])
# def delete_attendance_log(log_id):
#     try:
#         log = AttendanceLog.query.get_or_404(log_id)
#         db.session.delete(log)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Attendance log deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/delete_all_attendance_logs', methods=['DELETE'])
# def delete_all_attendance_logs():
#     try:
#         AttendanceLog.query.delete()
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'All attendance logs deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/attendance_logs_page', methods=['GET'])
# def attendance_logs_page():
#     return render_template('attendance_logs.html')

# # Camera Feed Management Routes
# @app.route('/add_camera_feed', methods=['POST'])
# def add_camera_feed():
#     name = request.form.get('name')
#     camera_url = request.form.get('camera_url')
#     camera_type = request.form.get('camera_type')
#     location = request.form.get('location')
#     description = request.form.get('description')

#     if not name or not camera_url or not camera_type:
#         return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

#     try:
#         camera_feed = CameraFeed(
#             name=name,
#             camera_url=camera_url,
#             camera_type=camera_type,
#             location=location,
#             description=description
#         )
#         db.session.add(camera_feed)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Camera feed added successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/camera_feeds', methods=['GET'])
# def list_camera_feeds():
#     camera_feeds = CameraFeed.query.all()
#     result = []
#     for cf in camera_feeds:
#         result.append({
#             'id': cf.id,
#             'name': cf.name,
#             'camera_url': cf.camera_url,
#             'camera_type': cf.camera_type,
#             'location': cf.location,
#             'description': cf.description,
#             'is_active': cf.is_active,
#             'created_at': cf.created_at.strftime('%Y-%m-%d %H:%M:%S')
#         })
#     return jsonify(result)

# @app.route('/edit_camera_feed/<int:camera_feed_id>', methods=['POST'])
# def edit_camera_feed(camera_feed_id):
#     camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
#     name = request.form.get('name')
#     camera_url = request.form.get('camera_url')
#     camera_type = request.form.get('camera_type')
#     location = request.form.get('location')
#     description = request.form.get('description')
#     is_active = request.form.get('is_active')

#     if name:
#         camera_feed.name = name
#     if camera_url:
#         camera_feed.camera_url = camera_url
#     if camera_type:
#         camera_feed.camera_type = camera_type
#     if location is not None:
#         camera_feed.location = location
#     if description is not None:
#         camera_feed.description = description
#     if is_active is not None:
#         camera_feed.is_active = is_active.lower() == 'true'

#     try:
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Camera feed updated successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/delete_camera_feed/<int:camera_feed_id>', methods=['POST'])
# def delete_camera_feed(camera_feed_id):
#     camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
#     try:
#         db.session.delete(camera_feed)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Camera feed deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/test_camera_feed/<int:camera_feed_id>', methods=['GET'])
# def test_camera_feed(camera_feed_id):
#     camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
#     camera = None
#     try:
#         if camera_feed.camera_type == 'device':
#             try:
#                 cam_index = int(camera_feed.camera_url)
#                 # Check if the device index is available
#                 test_cam = cv2.VideoCapture(cam_index)
#                 if not test_cam.isOpened():
#                     return jsonify({'status': 'error', 'message': f'Camera device index {cam_index} not available'}), 400
#                 test_cam.release()
#                 camera = cv2.VideoCapture(cam_index)
#             except ValueError:
#                 # Try to find an available camera
#                 camera = find_available_camera()
#                 if camera is None:
#                     return jsonify({'status': 'error', 'message': 'No camera devices available'}), 400
#         else:
#             camera = cv2.VideoCapture(camera_feed.camera_url, cv2.CAP_FFMPEG)

#         if camera is None or not camera.isOpened():
#             return jsonify({'status': 'error', 'message': 'Cannot connect to camera feed'}), 400

#         ret, frame = camera.read()
#         if ret and frame is not None:
#             return jsonify({'status': 'success', 'message': 'Camera feed is working'})
#         else:
#             return jsonify({'status': 'error', 'message': 'Camera feed is not working - no video signal'}), 400

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': f'Error testing camera: {str(e)}'}), 500
#     finally:
#         if camera is not None:
#             camera.release()

# @app.route('/add_camera_feed_form', methods=['GET'])
# def add_camera_feed_form():
#     return render_template('add_camera_feed.html')

# @app.route('/camera_feeds_page', methods=['GET'])
# def camera_feeds_page():
#     return render_template('camera_feeds.html')

# @app.route('/cctv_detection_page', methods=['GET'])
# def cctv_detection_page():
#     return render_template('cctv_detection.html')

# @app.route('/detection_config', methods=['GET'])
# def get_detection_config():
#     return jsonify({
#         'cooldown_seconds': app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
#     })

# @app.route('/employee_status_page', methods=['GET'])
# def employee_status_page():
#     return render_template('employee_status.html')

# @app.route('/employee_status_data', methods=['GET'])
# def get_employee_status_data():
#     """Get real-time employee status data"""
#     try:
#         # Get all employees
#         employees = Employee.query.all()
#         employee_data = []

#         # Get recent attendance logs
#         cooldown_seconds = app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
#         recent_time = datetime.utcnow() - timedelta(seconds=cooldown_seconds)

#         recent_attendances = AttendanceLog.query.filter(
#             AttendanceLog.timestamp >= recent_time
#         ).all()

#         # Create a map of recent detections
#         recent_attendance_map = {}
#         for attendance in recent_attendances:
#             if attendance.employee_name not in recent_attendance_map:
#                 recent_attendance_map[attendance.employee_name] = []
#             recent_attendance_map[attendance.employee_name].append({
#                 'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                 'attendance_type': attendance.attendance_type,
#                 'camera_feed_name': attendance.camera_feed_name
#             })

#         # Build status data for each employee
#         for employee in employees:
#             recent_attendances_for_employee = recent_attendance_map.get(employee.name, [])
#             is_present = len(recent_attendances_for_employee) > 0

#             employee_data.append({
#                 'id': employee.id,
#                 'name': employee.name,
#                 'employee_id': employee.employee_id,
#                 'image_url': url_for('static', filename=f'uploads/{employee.image_filename}', _external=True),
#                 'status': 'present' if is_present else 'absent',
#                 'attendance_count': len(recent_attendances_for_employee),
#                 'last_detected': recent_attendances_for_employee[-1]['timestamp'] if recent_attendances_for_employee else None,
#                 'recent_attendances': recent_attendances_for_employee
#             })

#         return jsonify({
#             'employees': employee_data,
#             'summary': {
#                 'total': len(employees),
#                 'present': len([e for e in employee_data if e['status'] == 'present']),
#                 'absent': len([e for e in employee_data if e['status'] == 'absent']),
#                 'attendance_rate': round((len([e for e in employee_data if e['status'] == 'present']) / len(employees)) * 100) if employees else 0,
#                 'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
#             }
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     socketio.run(app, debug=True) # host='0.0.0.0', port=5000, use_reloader=False

# # --- DEBUGGING PATCH START ---
# @app.route('/test_arcface', methods=['POST'])
# def test_arcface():
#     file = request.files.get('image')
#     if not file or not allowed_file(file.filename):
#         return jsonify({'status': 'error', 'message': 'Missing or invalid image'}), 400
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#     img = cv2.imread(filepath)
#     if img is None:
#         return jsonify({'status': 'error', 'message': 'Could not read uploaded image'}), 400
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     try:
#         faces = arcface_app.get(rgb_img)
#         if faces and hasattr(faces[0], 'embedding'):
#             return jsonify({'status': 'success', 'message': 'ArcFace embedding extracted', 'embedding': faces[0].embedding.tolist()})
#         else:
#             return jsonify({'status': 'error', 'message': 'ArcFace did not detect a face or embedding'}), 400
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': f'ArcFace exception: {str(e)}'}), 500

# # Patch extract_arcface_embedding_from_crop for more debug info
# import functools
# old_extract = extract_arcface_embedding_from_crop

# def debug_extract_arcface_embedding_from_crop(face_crop):
#     print(f"[DEBUG] Input to ArcFace: shape={face_crop.shape}, dtype={face_crop.dtype}, min={face_crop.min()}, max={face_crop.max()}")
#     try:
#         result = old_extract(face_crop)
#         if result is None:
#             print("[DEBUG] ArcFace embedding is None for this crop.")
#         else:
#             print(f"[DEBUG] ArcFace embedding shape: {np.array(result).shape}")
#         return result
#     except Exception as e:
#         print(f"[ERROR] Exception in ArcFace embedding extraction: {e}")
#         return None
# extract_arcface_embedding_from_crop = debug_extract_arcface_embedding_from_crop
# # --- DEBUGGING PATCH END ---



# # import eventlet
# # eventlet.monkey_patch()

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

# from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, Response
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.utils import secure_filename
# from PIL import Image
# import numpy as np
# import cv2
# import time
# from datetime import datetime, timedelta
# import threading
# import dlib
# import urllib.parse
# from mtcnn import MTCNN
# from insightface.app import FaceAnalysis
# from flask_migrate import Migrate
# import onnxruntime as ort
# from insightface.model_zoo import ArcFaceONNX
# from collections import deque
# import base64

# # Import the new FaceTracker
# from face_tracker import FaceTracker

# print("OpenCV version:", cv2.__version__)
# build_info = cv2.getBuildInformation()
# ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]
# print("FFMPEG in build info:", ffmpeg_support)

# # FFMPEG backend will be used explicitly when opening cameras
# print("FFMPEG backend will be used for RTSP streams")

# print(ort.get_device())

# session = ort.InferenceSession("insightface_repo/model_zoo/model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# print(session.get_providers())

# model = ArcFaceONNX("insightface_repo/model_zoo/model.onnx")
# model.prepare(ctx_id=0)
# mtcnn_detector = MTCNN()

# if getattr(dlib, 'DLIB_USE_CUDA', False):
#     print("Running on GPU")
# else:
#     print("Running on CPU")

# app = Flask(__name__)

# # Configurations
# app.config['SECRET_KEY'] = 'your_secret_key_here'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
# app.config['DETECTION_COOLDOWN_SECONDS'] = 30  # Time window to prevent duplicate detections

# # Ensure upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Initialize SocketIO
# from flask_socketio import SocketIO
# socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# db = SQLAlchemy(app)
# migrate = Migrate(app, db)

# class Employee(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     employee_id = db.Column(db.String(50), unique=True, nullable=False)
#     description = db.Column(db.Text, nullable=True)
#     image_filename = db.Column(db.String(200), nullable=False)
#     arcface_embedding = db.Column(db.PickleType, nullable=True)

#     def __repr__(self):
#         return f'<Employee {self.name}>'

# class CameraFeed(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     camera_url = db.Column(db.String(500), nullable=False)
#     camera_type = db.Column(db.String(50), nullable=False)
#     location = db.Column(db.String(200), nullable=True)
#     description = db.Column(db.Text, nullable=True)
#     is_active = db.Column(db.Boolean, default=True)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f'<CameraFeed {self.name}>'

# class AttendanceLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     employee_name = db.Column(db.String(100), nullable=False)
#     attendance_type = db.Column(db.String(20), nullable=False)
#     camera_feed_id = db.Column(db.Integer, db.ForeignKey('camera_feed.id'), nullable=True)
#     camera_feed_name = db.Column(db.String(100), nullable=True)
#     confidence_score = db.Column(db.Float, nullable=True)

#     def __repr__(self):
#         return f'<AttendanceLog {self.employee_name} at {self.timestamp}>'

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Initialize ArcFace model globally
# ctx_id = -1
# arcface_app = FaceAnalysis(name='buffalo_l')
# arcface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

# def extract_arcface_embedding_from_crop(face_crop):
#     if face_crop.shape[2] == 3:
#         rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
#     else:
#         rgb_crop = face_crop
#     faces = arcface_app.get(rgb_crop)
#     if faces:
#         return faces[0].embedding
#     return None

# def align_face_by_keypoints(img, keypoints, output_size=(112, 112)):
#     ref_pts = np.array([
#         [30.2946, 51.6963],
#         [65.5318, 51.5014],
#         [48.0252, 71.7366],
#         [33.5493, 92.3655],
#         [62.7299, 92.2041]], dtype=np.float32)
#     src_pts = np.array([
#         keypoints['left_eye'],
#         keypoints['right_eye'],
#         keypoints['nose'],
#         keypoints['mouth_left'],
#         keypoints['mouth_right']], dtype=np.float32)
#     from cv2 import estimateAffinePartial2D, warpAffine
#     M, _ = estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
#     aligned_face = warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=0)
#     return aligned_face

# def extract_arcface_embedding(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         return None, "Could not read image"
#     detections = mtcnn_detector.detect_faces(img)
#     if not detections:
#         return None, "No face detected by MTCNN"
#     first_detection = detections[0]
#     keypoints = first_detection['keypoints']
#     aligned_face = align_face_by_keypoints(img, keypoints, output_size=(112, 112))
#     embedding = extract_arcface_embedding_from_crop(aligned_face)
#     if embedding is not None:
#         return embedding, 'Embedding extracted successfully.'
#     return None, 'ArcFace failed on the aligned crop.'

# @app.route('/')
# def home():
#     return redirect('/live_detection_page')

# @app.route('/add_employee', methods=['POST'])
# def add_employee():
#     name = request.form.get('name')
#     employee_id = request.form.get('employee_id')
#     description = request.form.get('description')
#     files = request.files.getlist('images')
#     if not name or not employee_id or not files:
#         return jsonify({'status': 'error', 'message': 'Missing or invalid data'}), 400

#     embeddings = []
#     saved_filename = None

#     for file in files:
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             embedding, debug_message = extract_arcface_embedding(filepath)
#             if embedding is not None:
#                 embeddings.append(embedding)
#                 if saved_filename is None:
#                     saved_filename = filename # Save the first valid image's name
#             else:
#                 print(f"Could not process {filename}: {debug_message}")

#     if not embeddings:
#         return jsonify({'status': 'error', 'message': 'Failed to detect a face in any of the uploaded images.'}), 400

#     # Calculate the average embedding
#     arcface_embedding = np.mean(embeddings, axis=0)

#     try:
#         employee = Employee(
#             name=name,
#             employee_id=employee_id,
#             description=description,
#             image_filename=saved_filename,
#             arcface_embedding=arcface_embedding
#         )
#         db.session.add(employee)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Employee added successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/employees', methods=['GET'])
# def list_employees():
#     employees = Employee.query.all()
#     return jsonify([{'id': emp.id, 'name': emp.name, 'employee_id': emp.employee_id} for emp in employees])

# @app.route('/edit_employee/<int:employee_id>', methods=['POST'])
# def edit_employee(employee_id):
#     try:
#         employee = Employee.query.get_or_404(employee_id)

#         # Update textual details
#         employee.name = request.form['name']
#         employee.employee_id = request.form['employee_id']
#         employee.description = request.form['description']

#         images = request.files.getlist('images')
#         new_embeddings = []

#         # Process new images if any are uploaded
#         if images and images[0].filename:
#             for image in images:
#                 if image and allowed_file(image.filename):
#                     # We don't need to save the new files, just extract embeddings
#                     # To save memory, process from stream
#                     in_memory_file = io.BytesIO()
#                     image.save(in_memory_file)
#                     in_memory_file.seek(0)
#                     img_array = np.frombuffer(in_memory_file.read(), np.uint8)
#                     img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#                     if img_cv is None: continue

#                     dets = mtcnn_detector.detect_faces(img_cv)
#                     if not dets: continue

#                     det = dets[0] # Assume first face
#                     embedding = None
#                     try:
#                         aligned_face = align_face_by_keypoints(img_cv, det['keypoints'])
#                         embedding = extract_arcface_embedding_from_crop(aligned_face)
#                     except Exception:
#                         x, y, w, h = det['box']
#                         face_crop = img_cv[y:y+h, x:x+w]
#                         if face_crop.size > 0:
#                             embedding = extract_arcface_embedding_from_crop(face_crop)
                    
#                     if embedding is not None:
#                         new_embeddings.append(embedding)

#             if new_embeddings:
#                 all_embeddings = []
#                 if employee.arcface_embedding is not None:
#                     # Convert existing embedding from bytes back to numpy array
#                     existing_embedding = np.frombuffer(employee.arcface_embedding, dtype=np.float32)
#                     all_embeddings.append(existing_embedding)
                
#                 all_embeddings.extend(new_embeddings)
                
#                 # Calculate new average and update
#                 avg_embedding = np.mean(all_embeddings, axis=0)
#                 employee.arcface_embedding = avg_embedding.tobytes()

#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Employee updated successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/delete_employee/<int:employee_id>', methods=['POST'])
# def delete_employee(employee_id):
#     employee = Employee.query.get_or_404(employee_id)
#     db.session.delete(employee)
#     db.session.commit()
#     return jsonify({'status': 'success'})

# def cosine_similarity(a, b):
#     a = np.asarray(a).flatten()
#     b = np.asarray(b).flatten()
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # --- Real-time Camera Processing --- #

# camera_threads = {}
# camera_locks = {}
# stop_flags = {}
# face_trackers = {}  # Dictionary to hold a tracker for each camera

# # @app.route('/start_camera/<int:camera_id>')
# # def start_camera(camera_id):
# #     if camera_id in camera_threads and camera_threads[camera_id].is_alive():
# #         return jsonify({'status': 'success', 'message': 'Camera already running'})
# #     camera_feed = CameraFeed.query.get(camera_id)
# #     if not camera_feed or not camera_feed.is_active:
# #         return jsonify({'status': 'error', 'message': 'Camera not found or is inactive'}), 404
# #     stop_flags[camera_id] = threading.Event()
# #     face_trackers[camera_id] = FaceTracker(max_disappeared=5, history_size=10)
# #     thread = threading.Thread(target=camera_worker, args=(camera_id, stop_flags[camera_id]))
# #     camera_threads[camera_id] = thread
# #     thread.start()
# #     return jsonify({'status': 'success', 'message': f'Camera {camera_id} started'})

# # @app.route('/stop_camera/<int:camera_id>')
# # def stop_camera(camera_id):
# #     if camera_id in stop_flags:
# #         stop_flags[camera_id].set()
# #         if camera_id in camera_threads:
# #             camera_threads[camera_id].join()
# #             del camera_threads[camera_id]
# #         return jsonify({'status': 'success', 'message': f'Camera {camera_id} stopped'})
# #     return jsonify({'status': 'error', 'message': 'Camera not running'})

# def camera_worker(camera_id, stop_event):
#     with app.app_context():
#         camera_feed = CameraFeed.query.get(camera_id)
#         if camera_feed.camera_type == 'device':
#             cap = cv2.VideoCapture(int(camera_feed.camera_url))
#         else:
#             cap = cv2.VideoCapture(camera_feed.camera_url, cv2.CAP_FFMPEG)
#         if not cap.isOpened():
#             print(f"[ERROR] Could not open camera {camera_id}")
#             return
#         employees = Employee.query.filter(Employee.arcface_embedding.isnot(None)).all()
#         known_arcface_embeddings = [emp.arcface_embedding for emp in employees]
#         known_names = [emp.name for emp in employees]
#         tracker = face_trackers[camera_id]
#         frame_skip = 2
#         frame_count = 0
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 time.sleep(1)
#                 continue
#             frame_count += 1
#             if frame_count % frame_skip != 0:
#                 # Draw previous stable boxes on skipped frames for smoother visuals
#                 stable_objects = tracker.get_stable_names()
#                 for obj_id, (box, name) in stable_objects.items():
#                     x, y, x2, y2 = box
#                     cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
#                     label = f"{name}"
#                     cv2.rectangle(frame, (x, y - 20), (x + len(label)*9, y), (0, 255, 0), -1)
#                     cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 socketio.emit(f'video_frame_{camera_id}', {'image': base64.b64encode(buffer).decode('utf-8')})
#                 continue
            
#             height, width, _ = frame.shape
#             scale = 320 / width
#             small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
#             rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#             detections = mtcnn_detector.detect_faces(rgb_small_frame)
#             boxes = []
#             names = []
#             for det in detections:
#                 if det['confidence'] < 0.90:
#                     continue
#                 x, y, w, h = [int(v / scale) for v in det['box']]
#                 x2, y2 = x + w, y + h
#                 embedding = None
#                 try:
#                     aligned_face = align_face_by_keypoints(rgb_small_frame, det['keypoints'])
#                     embedding = extract_arcface_embedding_from_crop(aligned_face)
#                 except Exception:
#                     face_crop = frame[y:y2, x:x2]
#                     if face_crop.size > 0:
#                         embedding = extract_arcface_embedding_from_crop(face_crop)
#                 name = "Unknown"
#                 if embedding is not None:
#                     sims = [cosine_similarity(embedding, kemb) for kemb in known_arcface_embeddings]
#                     if sims and max(sims) > 0.5:
#                         name = known_names[sims.index(max(sims))]
#                 boxes.append((x, y, x2, y2))
#                 names.append(name)
            
#             stable_objects = tracker.update(boxes, names)
#             for obj_id, (box, name) in stable_objects.items():
#                 x, y, x2, y2 = box
#                 cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
#                 label = f"{name}"
#                 cv2.rectangle(frame, (x, y - 20), (x + len(label)*9, y), (0, 255, 0), -1)
#                 cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             _, buffer = cv2.imencode('.jpg', frame)
#             socketio.emit(f'video_frame_{camera_id}', {'image': base64.b64encode(buffer).decode('utf-8')})
#         cap.release()

# # --- Web Pages --- #
# # @app.route('/live_detection')
# # def live_detection():
# #     video_filename = request.args.get('video')
# #     camera_feed_id = request.args.get('camera_feed_id')
# #     if camera_feed_id:
# #         camera_feed_id = int(camera_feed_id)
# #     return app.response_class(gen_frames(video_filename, camera_feed_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/add_employee_form', methods=['GET'])
# def add_employee_form():
#     return render_template('add_employee.html')

# @app.route('/employees_page', methods=['GET'])
# def employees_page():
#     return render_template('employees.html')

# @app.route('/live_detection_page', methods=['GET'])
# def live_detection_page():
#     cameras = CameraFeed.query.filter_by(is_active=True).all()
#     return render_template('live_detection.html', cameras=cameras)


# @app.route('/attendance_logs', methods=['GET'])
# def attendance_logs():
#     logs = AttendanceLog.query.order_by(AttendanceLog.timestamp.desc()).all()
#     # Get all employees for image lookup, using trimmed, lowercased names as keys
#     employees = {e.name.strip().lower(): e for e in Employee.query.all()}
#     default_image = 'default.jpg'
#     return jsonify([
#         {
#             'id': log.id,
#             'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#             'employee_name': log.employee_name,
#             'attendance_type': log.attendance_type,
#             'camera_feed_name': log.camera_feed_name,
#             # Refactored image_url assignment for clarity and efficiency
#             'image_url': (lambda employee: url_for('static', filename=f'uploads/{employee.image_filename if employee else default_image}', _external=True))(
#                 employees.get((log.employee_name or '').strip().lower())
#             )
#         } for log in logs
#     ])

# @app.route('/delete_attendance_log/<int:log_id>', methods=['DELETE'])
# def delete_attendance_log(log_id):
#     try:
#         log = AttendanceLog.query.get_or_404(log_id)
#         db.session.delete(log)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Attendance log deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/delete_all_attendance_logs', methods=['DELETE'])
# def delete_all_attendance_logs():
#     try:
#         AttendanceLog.query.delete()
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'All attendance logs deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/attendance_logs_page', methods=['GET'])
# def attendance_logs_page():
#     return render_template('attendance_logs.html')

# # --- API Endpoints for Camera/Employee Management (remains the same) --- #
# # Camera Feed Management Routes
# @app.route('/add_camera_feed', methods=['POST'])
# def add_camera_feed():
#     name = request.form.get('name')
#     camera_url = request.form.get('camera_url')
#     camera_type = request.form.get('camera_type')
#     location = request.form.get('location')
#     description = request.form.get('description')

#     if not name or not camera_url or not camera_type:
#         return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

#     try:
#         camera_feed = CameraFeed(
#             name=name,
#             camera_url=camera_url,
#             camera_type=camera_type,
#             location=location,
#             description=description
#         )
#         db.session.add(camera_feed)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Camera feed added successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# # Add camera feed
# @app.route('/camera_feeds', methods=['GET'])
# def list_camera_feeds():
#     camera_feeds = CameraFeed.query.all()
#     result = []
#     for cf in camera_feeds:
#         result.append({
#             'id': cf.id,
#             'name': cf.name,
#             'camera_url': cf.camera_url,
#             'camera_type': cf.camera_type,
#             'location': cf.location,
#             'description': cf.description,
#             'is_active': cf.is_active,
#             'created_at': cf.created_at.strftime('%Y-%m-%d %H:%M:%S')
#         })
#     return jsonify(result)

# @app.route('/edit_camera_feed/<int:camera_feed_id>', methods=['POST'])
# def edit_camera_feed(camera_feed_id):
#     camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
#     name = request.form.get('name')
#     camera_url = request.form.get('camera_url')
#     camera_type = request.form.get('camera_type')
#     location = request.form.get('location')
#     description = request.form.get('description')
#     is_active = request.form.get('is_active')

#     if name:
#         camera_feed.name = name
#     if camera_url:
#         camera_feed.camera_url = camera_url
#     if camera_type:
#         camera_feed.camera_type = camera_type
#     if location is not None:
#         camera_feed.location = location
#     if description is not None:
#         camera_feed.description = description
#     if is_active is not None:
#         camera_feed.is_active = is_active.lower() == 'true'

#     try:
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Camera feed updated successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/delete_camera_feed/<int:camera_feed_id>', methods=['POST'])
# def delete_camera_feed(camera_feed_id):
#     camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
#     try:
#         db.session.delete(camera_feed)
#         db.session.commit()
#         return jsonify({'status': 'success', 'message': 'Camera feed deleted successfully'})
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/test_camera_feed/<int:camera_feed_id>', methods=['GET'])
# def test_camera_feed(camera_feed_id):
#     camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
#     camera = None
#     try:
#         if camera_feed.camera_type == 'device':
#             try:
#                 cam_index = int(camera_feed.camera_url)
#                 # Check if the device index is available
#                 test_cam = cv2.VideoCapture(cam_index)
#                 if not test_cam.isOpened():
#                     return jsonify({'status': 'error', 'message': f'Camera device index {cam_index} not available'}), 400
#                 test_cam.release()
#                 camera = cv2.VideoCapture(cam_index)
#             except ValueError:
#                 # Try to find an available camera
#                 camera = find_available_camera()
#                 if camera is None:
#                     return jsonify({'status': 'error', 'message': 'No camera devices available'}), 400
#         else:
#             camera = cv2.VideoCapture(camera_feed.camera_url, cv2.CAP_FFMPEG)

#         if camera is None or not camera.isOpened():
#             return jsonify({'status': 'error', 'message': 'Cannot connect to camera feed'}), 400

#         ret, frame = camera.read()
#         if ret and frame is not None:
#             return jsonify({'status': 'success', 'message': 'Camera feed is working'})
#         else:
#             return jsonify({'status': 'error', 'message': 'Camera feed is not working - no video signal'}), 400

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': f'Error testing camera: {str(e)}'}), 500
#     finally:
#         if camera is not None:
#             camera.release()

# @app.route('/add_camera_feed_form', methods=['GET'])
# def add_camera_feed_form():
#     return render_template('add_camera_feed.html')

# @app.route('/camera_feeds_page')
# def camera_feeds_page():
#     feeds = CameraFeed.query.all()
#     return render_template('camera_feeds.html', feeds=feeds)

# @app.route('/cctv_detection_page', methods=['GET'])
# def cctv_detection_page():
#     return render_template('cctv_detection.html')

# @app.route('/detection_config', methods=['GET'])
# def get_detection_config():
#     return jsonify({
#         'cooldown_seconds': app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
#     })

# @app.route('/employee_status_page', methods=['GET'])
# def employee_status_page():
#     return render_template('employee_status.html')

# @app.route('/employee_status_data', methods=['GET'])
# def get_employee_status_data():
#     """Get real-time employee status data"""
#     try:
#         # Get all employees
#         employees = Employee.query.all()
#         employee_data = []

#         # Get recent attendance logs
#         cooldown_seconds = app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
#         recent_time = datetime.utcnow() - timedelta(seconds=cooldown_seconds)

#         recent_attendances = AttendanceLog.query.filter(
#             AttendanceLog.timestamp >= recent_time
#         ).all()

#         # Create a map of recent detections
#         recent_attendance_map = {}
#         for attendance in recent_attendances:
#             if attendance.employee_name not in recent_attendance_map:
#                 recent_attendance_map[attendance.employee_name] = []
#             recent_attendance_map[attendance.employee_name].append({
#                 'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                 'attendance_type': attendance.attendance_type,
#                 'camera_feed_name': attendance.camera_feed_name
#             })

#         # Build status data for each employee
#         for employee in employees:
#             recent_attendances_for_employee = recent_attendance_map.get(employee.name, [])
#             is_present = len(recent_attendances_for_employee) > 0

#             employee_data.append({
#                 'id': employee.id,
#                 'name': employee.name,
#                 'employee_id': employee.employee_id,
#                 'image_url': url_for('static', filename=f'uploads/{employee.image_filename}', _external=True),
#                 'status': 'present' if is_present else 'absent',
#                 'attendance_count': len(recent_attendances_for_employee),
#                 'last_detected': recent_attendances_for_employee[-1]['timestamp'] if recent_attendances_for_employee else None,
#                 'recent_attendances': recent_attendances_for_employee
#             })

#         return jsonify({
#             'employees': employee_data,
#             'summary': {
#                 'total': len(employees),
#                 'present': len([e for e in employee_data if e['status'] == 'present']),
#                 'absent': len([e for e in employee_data if e['status'] == 'absent']),
#                 'attendance_rate': round((len([e for e in employee_data if e['status'] == 'present']) / len(employees)) * 100) if employees else 0,
#                 'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
#             }
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     socketio.run(app, debug=True, use_reloader=False)