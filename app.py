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
mtcnn_detector = MTCNN()
from insightface.app import FaceAnalysis
from flask_migrate import Migrate

print("OpenCV version:", cv2.__version__)
build_info = cv2.getBuildInformation()
ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]
print("FFMPEG in build info:", ffmpeg_support)

# FFMPEG backend will be used explicitly when opening cameras
print("FFMPEG backend will be used for RTSP streams")

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

# Add a basic SocketIO event handler
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('server_response', {'data': 'Connected to server'})

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

def check_ffmpeg_support():
    import cv2
    build_info = cv2.getBuildInformation()
    # Check for FFMPEG support (can be "FFMPEG: YES" or "FFMPEG: YES (prebuilt binaries)")
    # The actual format is "      FFMPEG:                      YES (prebuilt binaries)"
    if 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]:
        print("[INFO] OpenCV FFMPEG support detected.")
        return True
    else:
        print("[ERROR] OpenCV is not built with FFMPEG support. RTSP streams will not work.")
        print("[SOLUTION] Install opencv-python-headless instead of opencv-python")
        print("[SOLUTION] Run: pip uninstall opencv-python && pip install opencv-python-headless")
        return False

def check_system_ffmpeg():
    """Check if FFMPEG is available on the system"""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("[INFO] System FFMPEG detected")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Check for local FFMPEG installation
    import os
    local_ffmpeg = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        try:
            result = subprocess.run([local_ffmpeg, '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("[INFO] Local FFMPEG detected")
                # Add to PATH for current session
                ffmpeg_bin = os.path.join(os.getcwd(), "ffmpeg", "bin")
                if ffmpeg_bin not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = f"{ffmpeg_bin};{os.environ.get('PATH', '')}"
                return True
        except subprocess.TimeoutExpired:
            pass
    
    print("[WARNING] FFMPEG not found on system. RTSP streams may not work properly.")
    print("[SOLUTION] Install FFMPEG or run the install_ffmpeg_windows.py script")
    return False

# Check both OpenCV FFMPEG support and system FFMPEG
opencv_ffmpeg_ok = check_ffmpeg_support()
system_ffmpeg_ok = check_system_ffmpeg()

if not opencv_ffmpeg_ok:
    print("[CRITICAL] OpenCV FFMPEG support is required for RTSP streams!")
    print("[ACTION] Please fix this before using RTSP camera feeds.")


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

@app.route('/')
def home():
    return redirect('/live_detection_page')

# Add a new column to Employee for ArcFace embedding (if not already present)
# If using Alembic or migrations, this should be handled there. For now, add in-memory only for demonstration.
if not hasattr(Employee, 'arcface_embedding'):
    from sqlalchemy import PickleType
    Employee.arcface_embedding = db.Column(PickleType, nullable=True)

# Initialize ArcFace model globally
arcface_app = FaceAnalysis(name='buffalo_l')
arcface_app.prepare(ctx_id=0, det_size=(640, 640))

# Helper to extract ArcFace embedding from a face crop
import cv2
import numpy as np

def extract_arcface_embedding_from_crop(face_crop):
    faces = arcface_app.get(face_crop)
    if faces and hasattr(faces[0], 'embedding'):
        return faces[0].embedding
    return None

def extract_arcface_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[DEBUG] Could not read image: {image_path}")
        return None

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Robust preprocessing: histogram equalization, denoising, sharpening
    try:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    except Exception as e:
        print(f"[DEBUG] Histogram equalization failed: {e}")

    try:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    except Exception as e:
        print(f"[DEBUG] Denoising failed: {e}")

    # Sharpening
    try:
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
    except Exception as e:
        print(f"[DEBUG] Sharpening failed: {e}")

    # Resize very large images for better detection
    max_dim = max(img.shape[:2])
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    # Resize very small images up to 300px minimum
    min_dim = min(img.shape[:2])
    if min_dim < 300:
        scale = 300.0 / min_dim
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        detections = mtcnn_detector.detect_faces(rgb_img)
    except Exception as e:
        print(f"[ERROR] MTCNN detection failed: {e}")
        return None
    print(f"[DEBUG] MTCNN detections: {detections}")
    if not detections:
        debug_path = image_path.replace('.jpg', '_debug.jpg').replace('.jpeg', '_debug.jpeg').replace('.png', '_debug.png')
        cv2.imwrite(debug_path, img)
        print(f"[DEBUG] No face detected. Saved debug image to: {debug_path}")
        return None
    # Use the first detected face
    x, y, w, h = detections[0]['box']
    x, y = max(0, x), max(0, y)
    face_crop = img[y:y+h, x:x+w]
    return extract_arcface_embedding_from_crop(face_crop)

@app.route('/add_employee', methods=['POST'])
def add_employee():
    name = request.form.get('name')
    employee_id = request.form.get('employee_id')
    description = request.form.get('description')
    file = request.files.get('image')

    if not name or not employee_id or file is None or not hasattr(file, 'filename') or not file.filename or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Missing or invalid data'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    arcface_embedding = extract_arcface_embedding(filepath)
    if arcface_embedding is None:
        import time
        for _ in range(5):
            try:
                os.remove(filepath)
                break
            except PermissionError:
                time.sleep(0.2)
        return jsonify({'status': 'error', 'message': 'No face detected in image'}), 400

    try:
        employee = Employee(
            name=name,
            employee_id=employee_id,
            description=description,
            image_filename=filename,
            arcface_embedding=arcface_embedding
        )
        db.session.add(employee)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Employee added successfully'})
    except Exception as e:
        if os.path.exists(filepath):
            import time
            for _ in range(5):
                try:
                    os.remove(filepath)
                    break
                except PermissionError:
                    time.sleep(0.2)
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
    employee = Employee.query.get_or_404(employee_id)
    name = request.form.get('name')
    new_employee_id = request.form.get('employee_id')
    description = request.form.get('description')
    file = request.files.get('image')

    if name:
        employee.name = name
    if new_employee_id:
        employee.employee_id = new_employee_id
    if description is not None:
        employee.description = description

    if file and hasattr(file, 'filename') and file.filename and allowed_file(file.filename):
        old_path = os.path.join(app.config['UPLOAD_FOLDER'], employee.image_filename)
        if os.path.exists(old_path):
            os.remove(old_path)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        arcface_embedding = extract_arcface_embedding(filepath)
        if arcface_embedding is None:
            os.remove(filepath)
            return jsonify({'status': 'error', 'message': 'No face detected in new image'}), 400
        employee.arcface_embedding = arcface_embedding
        employee.image_filename = filename

    try:
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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if file is None or not hasattr(file, 'filename') or not file.filename:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Do not process video here, just return filename for live detection
    return jsonify({'status': 'success', 'video_filename': filename})

def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    if a is None or b is None:
        return 0.0
    return float(dot(a, b) / (norm(a) * norm(b)))

# Update gen_frames to use MTCNN for detection and ArcFace for recognition only
def gen_frames(video_filename=None, camera_feed_id=None):
    ctx = app.app_context()
    ctx.push()
    camera = None
    frame_queue = queue.Queue(maxsize=1)
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

        def camera_worker():
            with app.app_context():
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
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    frame_queue.put(error_frame)
                    return

                while not stop_event.is_set():
                    success, frame = camera.read()
                    if not success or frame is None:
                        time.sleep(0.01)
                        continue
                    frame_count += 1
                    process_this_frame = (frame_count % process_every_n == 0)
                    if process_this_frame:
                        # Resize for faster detection
                        scale = 0.5
                        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        detections = mtcnn_detector.detect_faces(rgb_small_frame)
                        height, width = frame.shape[:2]
                        boxes = []
                        names = []
                        confidences = []
                        for det in detections:
                            x, y, w, h = det['box']
                            # Scale coordinates back up
                            x = int(x / scale)
                            y = int(y / scale)
                            w = int(w / scale)
                            h = int(h / scale)
                            # Clamp to image boundaries
                            x = max(0, x)
                            y = max(0, y)
                            x2 = min(x + w, width)
                            y2 = min(y + h, height)
                            if w < 30 or h < 30:
                                continue  # Skip tiny detections
                            face_crop = frame[y:y2, x:x2]
                            embedding = extract_arcface_embedding_from_crop(face_crop)
                            name = "Unknown"
                            confidence = 0.0
                            if embedding is not None:
                                sims = [cosine_similarity(embedding, kemb) for kemb in known_arcface_embeddings]
                                if sims and max(sims) > 0.4:
                                    idx = sims.index(max(sims))
                                    name = known_names[idx]
                                    confidence = max(sims)
                            boxes.append((x, y, x2, y2))
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
                    frame_queue.put(frame)

        worker_thread = threading.Thread(target=camera_worker, daemon=True)
        worker_thread.start()

        while True:
            try:
                frame = frame_queue.get(timeout=2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                error_frame = generate_error_frame("Waiting for camera...")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Error in gen_frames: {e}")
        error_frame = generate_error_frame(f"Error: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        stop_event.set()
        if camera is not None:
            camera.release()
        ctx.pop()

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

@app.route('/live_detection')
def live_detection():
    video_filename = request.args.get('video')
    camera_feed_id = request.args.get('camera_feed_id')
    if camera_feed_id:
        camera_feed_id = int(camera_feed_id)
    return app.response_class(gen_frames(video_filename, camera_feed_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_employee_form', methods=['GET'])
def add_employee_form():
    return render_template('add_employee.html')

@app.route('/employees_page', methods=['GET'])
def employees_page():
    return render_template('employees.html')

@app.route('/live_detection_page', methods=['GET'])
def live_detection_page():
    return render_template('live_detection.html')

@app.route('/upload_video_page', methods=['GET'])
def upload_video_page():
    return render_template('upload_video.html')

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

@app.route('/camera_feeds_page', methods=['GET'])
def camera_feeds_page():
    return render_template('camera_feeds.html')

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
    socketio.run(app, debug=True) # host='0.0.0.0', port=5000, use_reloader=False