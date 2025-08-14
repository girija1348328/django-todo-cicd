import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
# Improve RTSP/FFmpeg capture stability and reduce decode noise
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;tcp|rtsp_flags;prefer_tcp|probesize;262144|analyzeduration;1000000|max_delay;5000000|buffer_size;1048576|loglevel;error'
)
import logging
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


from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
import asyncio

logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)


class SharedCameraManager:
    def __init__(self):
        self.cameras = {} 
        self.ref_counts = {}
        self.lock = threading.Lock()

    def _configure_capture(self, cap):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    
    def get_camera(self, camera_url):
        with self.lock:
            if camera_url not in self.cameras:
                cap = cv2.VideoCapture(int(camera_url) if str(camera_url).isdigit() else camera_url, cv2.CAP_FFMPEG)
                cap = self._configure_capture(cap)
                
                if cap.isOpened():
                    self.cameras[camera_url] = cap
                    self.ref_counts[camera_url] = 1
                else:
                    return None
            else:
                self.ref_counts[camera_url] += 1
            return self.cameras[camera_url]
    
    def release_camera(self, camera_url):
        with self.lock:
            if camera_url in self.ref_counts:
                self.ref_counts[camera_url] -= 1
                if self.ref_counts[camera_url] <= 0:
                    if camera_url in self.cameras:
                        self.cameras[camera_url].release()
                        del self.cameras[camera_url]
                    del self.ref_counts[camera_url]

shared_camera_manager = SharedCameraManager()

class WebRTCVideoStreamTrack(VideoStreamTrack):
    def __init__(self, camera_url):
        super().__init__()
        self.camera_url = camera_url
        self.running = True
        self.frame_count = 0

    async def recv(self):
        import time
        import traceback as tb
        frame_start = time.time()
        if not self.running:
            return None
        try:
            from av import VideoFrame
            import numpy as np
            from app_folder.models.employee import Employee
            employees = Employee.query.all()
            known_arcface_embeddings = []
            known_names = []
            for e in employees:
                img_embeddings = [img.arcface_embedding for img in e.images if img.arcface_embedding is not None]
                if img_embeddings:
                    avg_embedding = np.mean(img_embeddings, axis=0)
                    known_arcface_embeddings.append(avg_embedding)
                    known_names.append(e.name)
            cap = shared_camera_manager.get_camera(self.camera_url)
            frame = get_model_output_frame(cap, arcface_app, known_arcface_embeddings, known_names)
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts, video_frame.time_base = self.next_timestamp(), 1/30
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                pass
            frame_end = time.time()
            return video_frame
        except Exception as e:
            try:
                import numpy as np
                import cv2
                from av import VideoFrame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Stream Error", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts, video_frame.time_base = self.next_timestamp(), 1/30
                return video_frame
            except Exception as e2:
                await asyncio.sleep(0.033)
                return None
        finally:
            pass

    def stop(self):
        self.running = False


webrtc_peer_connections = {}

FACE_DETECTION_CONFIDENCE = 0.95  
FACE_RECOGNITION_THRESHOLD = 0.6  
MIN_FACE_SIZE_PX = 60  
MAX_YAW_ANGLE_DEG = 30  
MIN_RECOGNITION_MARGIN = 0.1  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_recognition_system")

build_info = cv2.getBuildInformation()
ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]

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

from datetime import datetime
server_start_time = datetime.utcnow()

def get_server_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['DETECTION_COOLDOWN_SECONDS'] = 30  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from flask_socketio import SocketIO
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

from flask_sock import Sock
sock = Sock(app)



websocket_streaming_active = False
websocket_stream_settings = {
    'overlay': True
}
connected_clients = {}
unique_clients = {}
stream_viewers = set()
stream_viewer_clients = {}

def rebuild_unique_clients():
    global unique_clients, stream_viewer_clients
    unique_clients = {}
    stream_viewer_clients = {}
    
    for sid, client_info in connected_clients.items():
        unique_key = f"{client_info['ip']}_{client_info['user_agent'][:20]}"
        
        if unique_key in unique_clients:
            unique_clients[unique_key]['session_count'] += 1
            if client_info['connected_time'] > unique_clients[unique_key]['connected_time']:
                unique_clients[unique_key]['connected_time'] = client_info['connected_time']
        else:
            unique_clients[unique_key] = {
                'ip': client_info['ip'],
                'user_agent': client_info['user_agent'],
                'connected_time': client_info['connected_time'],
                'session_count': 1
            }
    
    for sid in stream_viewers:
        if sid in connected_clients:
            client_info = connected_clients[sid]
            unique_key = f"{client_info['ip']}_{client_info['user_agent'][:20]}"
            
            if unique_key in stream_viewer_clients:
                stream_viewer_clients[unique_key]['session_count'] += 1
                if client_info['connected_time'] > stream_viewer_clients[unique_key]['connected_time']:
                    stream_viewer_clients[unique_key]['connected_time'] = client_info['connected_time']
            else:
                stream_viewer_clients[unique_key] = {
                    'ip': client_info['ip'],
                    'user_agent': client_info['user_agent'],
                    'connected_time': client_info['connected_time'],
                    'session_count': 1
                }

@socketio.on('connect')
def handle_connect():
    client_info = {
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown')[:50],
        'connected_time': datetime.utcnow().strftime('%H:%M:%S')
    }
    
    connected_clients[request.sid] = client_info
    
    rebuild_unique_clients()
    
    print(f"[CONNECT] Client connected: {client_info['ip']}, sid: {request.sid} (Total unique clients: {len(unique_clients)})")
    
    socketio.emit('client_count_update', {
        'count': len(unique_clients),
        'clients': list(unique_clients.values())
    })
    
    socketio.emit('stream_viewer_count_update', {
        'count': len(stream_viewer_clients),
        'clients': list(stream_viewer_clients.values())
    })

@socketio.on('disconnect')
def handle_disconnect():
    global websocket_streaming_active
    if request.sid in connected_clients:
        client_info = connected_clients.pop(request.sid)
    
    stream_viewers.discard(request.sid)
    
    if not stream_viewers and websocket_streaming_active:
        websocket_streaming_active = False
    
    rebuild_unique_clients()
    
    socketio.emit('client_count_update', {
        'count': len(unique_clients),
        'clients': list(unique_clients.values())
    })
    
    socketio.emit('stream_viewer_count_update', {
        'count': len(stream_viewer_clients),
        'clients': list(stream_viewer_clients.values())
    })

@socketio.on('test_event')
def handle_test_event(data):
    print(f"[TEST_EVENT] Received test event from {request.remote_addr}, sid: {request.sid}, data: {data}")
    socketio.emit('test_response', {'message': 'Backend received your test event!'}, room=request.sid)
    return {'status': 'received'}

@socketio.on('test_detection_event')
def handle_test_detection_event():
    print(f"[TEST_DETECTION] Manually triggering test detection event from {request.remote_addr}")
    sid = request.sid
    test_detection = {
        'employee_name': 'Test Employee',
        'confidence': 0.95,
        'designation': 'Test Role',
        'department': 'Test Dept',
        'location': 'Test Location',
        'timestamp': datetime.utcnow().strftime('%H:%M:%S'),
        'access': True,
        'camera_name': 'Test Camera',
        'camera_id': 999,
        'image_url': url_for('static', filename='img/default-avatar.png', _external=True)
    }
    print("[TEST_DETECTION] Emitting test detection_event with data:", test_detection)
    # Emit to the triggering client explicitly, and also broadcast globally
    socketio.emit('detection_event', test_detection, room=sid)
    socketio.emit('detection_event', test_detection)
    print("[TEST_DETECTION] Test detection_event emitted successfully to client and broadcast")
    return {'status': 'test_detection_emitted'}

@socketio.on('join_stream_viewer')
def handle_join_stream_viewer():
    global websocket_streaming_active
    print(f"[JOIN_STREAM_VIEWER] Handler called by {request.remote_addr}, sid: {request.sid}")
    stream_viewers.add(request.sid)
    viewer_count = len(stream_viewers)
    print(f"[JOIN_STREAM_VIEWER] Client joined stream viewer: {request.remote_addr}, sid: {request.sid} (Total viewers: {viewer_count})")
    print(f"[JOIN_STREAM_VIEWER] stream_viewers set now contains: {stream_viewers}")
    
    if not websocket_streaming_active:
        websocket_streaming_active = True
        print(f"[JOIN_STREAM_VIEWER] WebSocket streaming activated (at least one viewer present)")
    else:
        print(f"[JOIN_STREAM_VIEWER] WebSocket streaming was already active")
        pass
    
    print(f"[JOIN_STREAM_VIEWER] Final state - websocket_streaming_active: {websocket_streaming_active}, stream_viewers: {stream_viewers}")
    
    rebuild_unique_clients()
    
    socketio.emit('stream_status', {
        'active': websocket_streaming_active,
        'settings': websocket_stream_settings
    }, room=request.sid)
    
    socketio.emit('stream_viewer_count_update', {
        'count': len(stream_viewer_clients),
        'clients': list(stream_viewer_clients.values())
    })

from flask import copy_current_request_context

@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    global webrtc_peer_connections, webrtc_video_track
    sid = request.sid
    offer = data.get('sdp')
    camera_url = data.get('camera_url', 0)
    
    
    if not offer:
        socketio.emit('webrtc_error', {'msg': 'No SDP offer received'}, room=sid)
        return
    
    @copy_current_request_context
    def run_webrtc_setup():
        async def setup_webrtc():
            try:
                pc = RTCPeerConnection(configuration=RTCConfiguration(
                    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
                ))

                webrtc_peer_connections[sid] = {'pc': pc, 'camera_url': camera_url}
                
                video_track = WebRTCVideoStreamTrack(camera_url)
                pc.addTrack(video_track)
                
                
                pc._video_track = video_track
                
                @pc.on('connectionstatechange')
                async def on_connectionstatechange():
                    if pc.connectionState in ('closed', 'failed', 'disconnected'):
                        if hasattr(pc, '_video_track'):
                            pc._video_track.stop()
                        
                        if sid in webrtc_peer_connections:
                            connection_info = webrtc_peer_connections.pop(sid)
                            shared_camera_manager.release_camera(connection_info['camera_url'])
                        
                        await pc.close()
                            
                @pc.on('iceconnectionstatechange')
                def on_iceconnectionstatechange():
                    pass
                
                await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type='offer'))
                
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                answer_payload = {
                    'sdp': pc.localDescription.sdp,
                    'type': pc.localDescription.type
                }
                try:
                    emit_result = socketio.emit('webrtc_answer', answer_payload, room=sid)
                except Exception as emit_exc:
                    pass
                
            except Exception as e:
                socketio.emit('webrtc_error', {'msg': f'WebRTC setup failed: {str(e)}'}, room=sid)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(setup_webrtc())
        except Exception as e:
            socketio.emit('webrtc_error', {'msg': f'Event loop setup failed: {str(e)}'}, room=sid)
        finally:
            if 'loop' in locals():
                loop.close()
    
    import threading
    webrtc_thread = threading.Thread(target=run_webrtc_setup, daemon=True)
    webrtc_thread.start()

@socketio.on('webrtc_ice_candidate')
def handle_webrtc_ice_candidate(data):
    """Handle ICE candidate from client."""
    sid = request.sid
    candidate = data.get('candidate')
    sdpMid = data.get('sdpMid')
    sdpMLineIndex = data.get('sdpMLineIndex')
    connection_info = webrtc_peer_connections.get(sid)
    
    
    if not connection_info:
        return
    if not candidate:
        return
    
    pc = connection_info['pc']
    
    @copy_current_request_context
    def add_ice():
        async def add():
            try:
                from aiortc import RTCIceCandidate
                parts = candidate.split()
                if len(parts) >= 8 and parts[0].startswith('candidate:'):
                    foundation = parts[0][10:]
                    component = int(parts[1])
                    protocol = parts[2].lower()
                    priority = int(parts[3])
                    ip = parts[4]
                    port = int(parts[5])
                    candidate_type = parts[7]
                    
                    ice_candidate = RTCIceCandidate(
                        component=component,
                        foundation=foundation,
                        ip=ip,
                        port=port,
                        priority=priority,
                        protocol=protocol,
                        type=candidate_type
                    )
                    ice_candidate.sdpMid = sdpMid
                    ice_candidate.sdpMLineIndex = sdpMLineIndex
                    await pc.addIceCandidate(ice_candidate)
                else:
                    pass
            except Exception as e:
                pass
        asyncio.run(add())
    add_ice()

@socketio.on('request_system_info')
def handle_system_info_request():
    try:
        import socket as sock
        hostname = sock.gethostname()
        server_ip = sock.gethostbyname(hostname)
    except:
        server_ip = '127.0.0.1'
    
    active_cameras = CameraFeed.query.filter_by(is_active=True).count()
    
    socketio.emit('system_info', {
        'server_ip': server_ip,
        'active_cameras': active_cameras,
        'uptime': 'Running'
    }, room=request.sid)

@socketio.on('refresh_stream')
def handle_refresh_stream():
    pass


def emit_websocket_frame(camera_id, camera_name, frame, detections=None, employee_name=None, confidence=0.0, 
                        designation='', department='', location=None, access=True, timestamp=None):
    """
    Unified function to handle all detection event emissions.
    Always include the correct employee image URL from the database in the detection_event.
    
    Args:
        camera_id: ID of the camera
        camera_name: Name of the camera
        frame: The video frame (unused in this function but kept for backward compatibility)
        detections: Optional dict containing detection details (legacy parameter)
        employee_name: Name of the detected employee
        confidence: Confidence score of the detection
        designation: Employee's designation
        department: Employee's department
        location: Detection location (defaults to camera_name if not provided)
        access: Whether access is granted
        timestamp: Detection timestamp (defaults to current time if not provided)
    """
    global websocket_streaming_active, websocket_stream_settings, stream_viewers
    
    # Always allow global emit; per-viewer emits happen if viewers exist
        
    try:
        # Handle legacy detections parameter
        if detections is not None and isinstance(detections, dict):
            # If detections is provided, use it to populate the fields
            if hasattr(emit_websocket_frame, 'access_override'):
                access = emit_websocket_frame.access_override
                del emit_websocket_frame.access_override
            else:
                access = detections.get('access', access)
                
            employee_name = detections.get('employee_name', employee_name)
            confidence = detections.get('confidence', confidence)
            designation = detections.get('designation', designation)
            department = detections.get('department', department)
            location = detections.get('location', location or camera_name)
            timestamp = detections.get('timestamp', timestamp)
        
        # Set default values if not provided
        now_str = datetime.utcnow().strftime('%H:%M:%S')
        timestamp = timestamp or now_str
        location = location or camera_name
        
        # If name not provided, treat as unknown without forcing fake details
        if employee_name is None or employee_name == '':
            employee_name = 'Unknown Visitor'
            access = False
            designation = ''
            department = ''
        
        # Lookup employee image from the database (ensure app context for url_for)
        from app_folder.models.employee import Employee
        from flask import url_for
        image_url = None
        try:
            with app.app_context():
                image_url = url_for('static', filename='img/default-avatar.png', _external=True)
                if employee_name and employee_name != 'Unknown Visitor':
                    employee = Employee.query.filter_by(name=employee_name).first()
                    if employee and employee.images:
                        image_url = url_for('static', filename=f'uploads/{employee.images[0].image_filename}', _external=True)
        except Exception:
            # Fallback if context/url_for fails
            image_url = image_url or ''

        # Construct the detection event
        detection = {
            'employee_name': employee_name,
            'confidence': round(float(confidence), 2),  # Ensure float and round to 2 decimal places
            'designation': designation,
            'department': department,
            'location': location,
            'timestamp': timestamp,
            'access': bool(access),
            'camera_name': camera_name,
            'camera_id': camera_id,
            'image_url': image_url
        }
        
        # Log the detection event
        print("[SOCKETIO] Emitting detection_event with fields:")
        logger.info("[SOCKETIO] Emitting detection_event with fields:")
        for k, v in detection.items():
            print(f"    {k}: {v}")
            logger.info(f"    {k}: {v}")
            
        # Emit the event (Flask-SocketIO for app UIs)
        try:
            socketio.emit('detection_event', detection)
            # Also emit to each active stream viewer room to ensure delivery to new/late subscribers
            try:
                for viewer_sid in list(stream_viewers):
                    socketio.emit('detection_event', detection, room=viewer_sid)
            except Exception:
                pass
            print("[SOCKETIO] detection_event emitted successfully.")
            logger.info("detection_event emitted successfully.")
            
            print(f"[DEBUG] stream_viewers set contains: {stream_viewers}")
            if stream_viewers:
                print(f"[SOCKETIO] Also emitting to {len(stream_viewers)} stream viewers")
                for viewer_sid in stream_viewers:
                    socketio.emit('detection_event', detection, room=viewer_sid)
            else:
                print("[DEBUG] No stream viewers found, emitting globally only")
                
        except Exception as emit_exc:
            print(f"[ERROR] Exception during socketio.emit: {emit_exc}")
            logger.error(f"Exception during socketio.emit: {emit_exc}")
        
        # Also enqueue for the test WebSocket server (8769) used by client.html
        try:
            detection_broadcast_queue.put_nowait(detection)
        except Exception:
            pass
            
    except Exception as e:
        print(f"[ERROR] Exception in emit_websocket_frame: {e}")
        logger.error(f"Exception in emit_websocket_frame: {e}")

db.init_app(app)
migrate = Migrate(app, db)

from app_folder.models.employee import Employee, EmployeeImage
from app_folder.models.camera_feed import CameraFeed

from app_folder.models.detection_history import DetectionHistory
from app_folder.routes.detection_history import detection_history_bp

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Register blueprints
app.register_blueprint(detection_history_bp)

def encode_rtsp_url(rtsp_url):
    if 'rtsp://' in rtsp_url:
        try:
            prefix, rest = rtsp_url.split('://', 1)
            # Split at the last '@' to support '@' inside password
            if '@' in rest:
                creds, path = rest.rsplit('@', 1)
                if ':' in creds:
                    user, pwd = creds.split(':', 1)
                    user_enc = urllib.parse.quote(user, safe='')
                    pwd_enc = urllib.parse.quote(pwd, safe='')
                    return f"{prefix}://{user_enc}:{pwd_enc}@{path}"
        except Exception as e:
            pass
    return rtsp_url

if not hasattr(Employee, 'arcface_embedding'):
    from sqlalchemy import PickleType
    Employee.arcface_embedding = db.Column(PickleType, nullable=True)

ctx_id = -1
    

def extract_arcface_embedding_from_crop(face_crop):
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
    ref_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    src_pts = np.array([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ], dtype=np.float32)

    from cv2 import estimateAffinePartial2D, warpAffine
    M, _ = estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
    aligned_face = warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=0)
    return aligned_face

def extract_arcface_embedding(image_path):
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

@app.route('/test_webrtc_stream')
def test_webrtc_stream():
    return render_template('test_webrtc_stream.html')

@app.route('/test_fallback_stream')
def test_fallback_stream():
    return render_template('test_fallback_stream.html')

@app.route('/test_events')
def test_events():
    return render_template('test_ws_event.html')

@app.route('/add_employee', methods=['POST'])
def add_employee():
    office_employee_id = request.form.get('office_employee_id')
    name = request.form.get('name')
    designation = request.form.get('designation')
    department = request.form.get('department')
    location = request.form.get('location')
    files = request.files.getlist('images')
    if not office_employee_id or not name or not designation or not department or not location or not files:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    embeddings = []
    image_filenames = []
    debug_msgs = []

    if Employee.query.filter_by(office_employee_id=office_employee_id).first():
        return jsonify({'status': 'error', 'message': 'Office Employee ID already exists'}), 400

    try:
        employee = Employee(
            office_employee_id=office_employee_id,
            name=name,
            designation=designation,
            department=department,
            location=location
        )
        db.session.add(employee)
        db.session.flush()

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
        debug_msgs.append(f"Added Employee: {employee.name} (office_employee_id={employee.office_employee_id}, id={employee.id})")
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
            'office_employee_id': e.office_employee_id,
            'name': e.name,
            'designation': e.designation,
            'department': e.department,
            'location': e.location,
            'image_url': image_url,
            'images': images
        })
    return jsonify(result)

@app.route('/edit_employee/<int:employee_id>', methods=['POST'])
def edit_employee(employee_id):
    try:
        employee = Employee.query.get_or_404(employee_id)
        new_office_id = request.form['office_employee_id']
        if new_office_id != employee.office_employee_id:
            if Employee.query.filter_by(office_employee_id=new_office_id).first():
                return jsonify({'status': 'error', 'message': 'Office Employee ID already exists'}), 400
            employee.office_employee_id = new_office_id
        employee.name = request.form['name']
        employee.designation = request.form['designation']
        employee.department = request.form['department']
        employee.location = request.form['location']
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

def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_employee(embedding, known_embeddings, known_names, threshold=0.5, top_n=3, min_margin=0.07):
    if embedding is None or not known_embeddings:
        return "Unknown", 0.0, {"reason": "No embedding or no known embeddings"}
    sims = [cosine_similarity(embedding, kemb) for kemb in known_embeddings]
    if not sims:
        return "Unknown", 0.0, {"reason": "No similarity scores"}
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    top_scores = [sims[i] for i in top_indices]
    top_names = [known_names[i] for i in top_indices]
    debug_info = {"all_scores": sims, "top_scores": top_scores, "top_names": top_names}
    if top_scores[0] > threshold:
        if len(top_scores) > 1 and top_n > 1 and (top_scores[0] - top_scores[1]) < min_margin:
            debug_info["reason"] = f"Top-1 margin too small: {top_scores[0] - top_scores[1]:.3f}"
            return "Unknown", top_scores[0], debug_info
        debug_info["reason"] = "Recognized"
        return top_names[0], top_scores[0], debug_info
    debug_info["reason"] = f"Top-1 below threshold: {top_scores[0]:.3f} < {threshold}"
    return "Unknown", top_scores[0], debug_info


camera_threads = {}
camera_locks = {}
stop_flags = {}
face_trackers = {}  
websocket_frame_queues = {}
last_websocket_frame_time = {}

DETECTION_CONFIDENCE_THRESHOLD = 0.85  
MIN_FACE_SIZE = 60  
MIN_ASPECT_RATIO = 0.8  
MAX_ASPECT_RATIO = 1.2
MIN_BLURRINESS = 50  
RECOGNITION_THRESHOLD = 0.6  
MIN_RECOGNITION_MARGIN = 0.1  
MAX_YAW_ANGLE = 30  
MAX_DETECTION_FAILURES = 3  

def get_model_output_frame(camera, arcface_app, known_arcface_embeddings, known_names):
    import numpy as np
    import cv2
    try:
        ret, frame = camera.read()
        if not ret or frame is None or (hasattr(frame, 'size') and frame.size == 0):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Signal", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        faces = arcface_app.get(frame)
        boxes, names, confidences = [], [], []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            embedding = face.embedding
            name, confidence = "Unknown", 0.0
            if embedding is not None:
                name, confidence, _ = recognize_employee(
                    embedding, known_arcface_embeddings, known_names, threshold=0.5, top_n=3, min_margin=0.07)
            boxes.append((x1, y1, x2, y2))
            names.append(name)
            confidences.append(confidence)
        for i, (box, name, confidence) in enumerate(zip(boxes, names, confidences)):
            try:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if name != "Unknown" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except IndexError as ie:
                print(f"[DRAW ERROR] IndexError for face #{i}: {ie}. Data: box={box}, name={name}")
            except Exception as draw_err:
                print(f"[DRAW ERROR] General error for face #{i} ({name}): {draw_err}")
        return frame
    except Exception as e:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Model error: {str(e)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

def gen_frames(video_filename=None, camera_feed_id=None):
    from flask import current_app
    with app.app_context():
        print("[DEBUG] gen_frames called with video_filename=", video_filename, "camera_feed_id=", camera_feed_id)
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
                    # Give the RTSP session a brief time to handshake and start
                    time.sleep(0.5)
                    retry_count = 0
                    max_retries = 5
                    while (camera is None or not camera.isOpened()) and retry_count < max_retries:
                        print(f"[WARN] Failed to open RTSP stream. Retrying {retry_count+1}/{max_retries}...")
                        time.sleep(1.5)
                        camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        time.sleep(0.5)
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
                process_every_n = 5  # Increased for smoother output
                last_boxes = []
                last_names = []
                last_confidences = []
                last_emit_times = {}

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
                        if not success or frame is None or (hasattr(frame, 'size') and frame.size == 0):
                            time.sleep(0.01)
                            continue
                        frame_count += 1
                        process_this_frame = (frame_count % process_every_n == 0)
                        if process_this_frame:
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
                            # Emit detection events aligned with box updates
                            try:
                                camera_name = camera_feed.name if camera_feed_id and 'camera_feed' in locals() and camera_feed else "Live Camera"
                            except Exception:
                                camera_name = "Live Camera"
                            from datetime import datetime
                            now_ts = time.time()
                            for name, confidence in zip(last_names, last_confidences):
                                key = name if name != "Unknown" else "Unknown"
                                last_ts = last_emit_times.get(key, 0)
                                if now_ts - last_ts < 1.0:
                                    continue  # simple cooldown per identity
                                timestamp = datetime.utcnow().strftime('%H:%M:%S')
                                try:
                                    if name != "Unknown" and confidence > 0.6:
                                        employee_obj = next((e for e in employees if e.name == name), None)
                                        if employee_obj:
                                            emit_websocket_frame(
                                                camera_id=camera_feed_id or 0,
                                                camera_name=camera_name,
                                                frame=frame.copy(),
                                                employee_name=employee_obj.name,
                                                confidence=confidence,
                                                designation=employee_obj.designation,
                                                department=employee_obj.department,
                                                location=employee_obj.location,
                                                access=True,
                                                timestamp=timestamp
                                            )
                                        else:
                                            emit_websocket_frame(
                                                camera_id=camera_feed_id or 0,
                                                camera_name=camera_name,
                                                frame=frame.copy(),
                                                employee_name=name,
                                                confidence=confidence,
                                                access=True,
                                                timestamp=timestamp
                                            )
                                    else:
                                        emit_websocket_frame(
                                            camera_id=camera_feed_id or 0,
                                            camera_name=camera_name,
                                            frame=frame.copy(),
                                            employee_name="Unknown Visitor",
                                            confidence=confidence,
                                            access=False,
                                            timestamp=timestamp
                                        )
                                    last_emit_times[key] = now_ts
                                except Exception as emit_err:
                                    print(f"[EMIT ERROR] Could not emit detection event: {emit_err}")
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
    for i in range(4):
        try:
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
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
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)

    (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)

    x = (frame.shape[1] - text_width) // 2
    y = (frame.shape[0] + text_height) // 2

    cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)

    help_text = "Check camera connection or try video upload"
    (help_width, help_height), _ = cv2.getTextSize(help_text, font, 0.7, 1)
    help_x = (frame.shape[1] - help_width) // 2
    help_y = y + 50
    cv2.putText(frame, help_text, (help_x, help_y), font, 0.7, (200, 200, 200), 1)

    return frame 

import websockets
import json

DEFAULT_CAMERA_FEED_ID = 1

connected_ws_clients = set()

def extract_jpeg_bytes(multipart_chunk):
    """Extract JPEG bytes from a multipart HTTP chunk."""
    if b'Content-Type: image/jpeg' in multipart_chunk:
        jpg_start = multipart_chunk.find(b'\r\n\r\n') + 4
        jpg_end = multipart_chunk.rfind(b'\r\n')
        return multipart_chunk[jpg_start:jpg_end]
    return None

async def ws_video_stream(websocket, path=None):
    connected_ws_clients.add(websocket)
    frame_generator = gen_frames(camera_feed_id=DEFAULT_CAMERA_FEED_ID)
    try:
        for frame_bytes in frame_generator:
            # Send current video frame
            jpg_data = extract_jpeg_bytes(frame_bytes)
            if jpg_data:
                try:
                    jpg_as_text = base64.b64encode(jpg_data).decode('utf-8')
                    message = json.dumps({"type": "video_frame", "frame": jpg_as_text})
                    await websocket.send(message)
                except Exception:
                    pass

            # Broadcast any pending employee details to all connected clients
            try:
                while True:
                    detection = detection_broadcast_queue.get_nowait()
                    payload = {
                        "type": "employee_details",
                        "name": detection.get("employee_name", "Unknown Visitor"),
                        "designation": detection.get("designation", ""),
                        "department": detection.get("department", ""),
                        "location": detection.get("location", ""),
                        "access": detection.get("access", False),
                        "confidence": detection.get("confidence", 0.0),
                        "timestamp": detection.get("timestamp", ""),
                        "camera_name": detection.get("camera_name", ""),
                        "camera_id": detection.get("camera_id", 0)
                    }
                    for client in list(connected_ws_clients):
                        try:
                            await client.send(json.dumps(payload))
                        except Exception:
                            try:
                                connected_ws_clients.remove(client)
                            except Exception:
                                pass
            except Exception:
                pass

            await asyncio.sleep(0.03)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        pass
    finally:
        try:
            connected_ws_clients.remove(websocket)
        except Exception:
            pass

def start_ws_server():
    async def ws_main():
        async with websockets.serve(ws_video_stream, "0.0.0.0", 8765):
            print("[WS] Unified WebSocket running at ws://0.0.0.0:8765")
            await asyncio.Future()
    def run_ws():
        asyncio.run(ws_main())
    t = threading.Thread(target=run_ws, daemon=True)
    t.start()

# Detection broadcast queue used by the unified WS server
detection_broadcast_queue = queue.Queue()

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







@app.route('/attendance_logs_page', methods=['GET'])
def attendance_logs_page():
    return render_template('attendance_logs.html')


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

def test_camera_feed(camera_feed_id):
    camera_feed = CameraFeed.query.get_or_404(camera_feed_id)
    camera = None
    try:
        if camera_feed.camera_type == 'device':
            try:
                cam_index = int(camera_feed.camera_url)
                test_cam = cv2.VideoCapture(cam_index)
                if not test_cam.isOpened():
                    return jsonify({'status': 'error', 'message': f'Camera device index {cam_index} not available'}), 400
                test_cam.release()
                camera = cv2.VideoCapture(cam_index)
            except ValueError:
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
        employees = Employee.query.all()
        employee_data = []

        cooldown_seconds = app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
        recent_time = datetime.utcnow() - timedelta(seconds=cooldown_seconds)

        recent_attendances = AttendanceLog.query.filter(
            AttendanceLog.timestamp >= recent_time
        ).all()

        recent_attendance_map = {}
        for attendance in recent_attendances:
            if attendance.employee_name not in recent_attendance_map:
                recent_attendance_map[attendance.employee_name] = []
            recent_attendance_map[attendance.employee_name].append({
                'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'attendance_type': attendance.attendance_type,
                'camera_feed_name': attendance.camera_feed_name
            })

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

camera_threads = {}
camera_active = {}
face_trackers = {}

def process_camera_feed(camera_id, camera_name, camera_url):
    """Process camera feed with face detection and recognition"""
    global websocket_streaming_active
    
    try:
        if camera_url.isdigit():
            cap = cv2.VideoCapture(int(camera_url))
        else:
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}: {camera_url}")
            return
        
        face_trackers[camera_id] = FaceTracker()
        
        print(f"[INFO] Started processing camera {camera_id}: {camera_name}")
        
        while camera_active.get(camera_id, False):
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
            
            detections = []
            try:
                if arcface_app is not None:
                    faces = arcface_app.get(frame)
                    
                    for face in faces:
                        embedding = face.embedding
                        if embedding is not None:
                            embedding = embedding / np.linalg.norm(embedding)
                            
                            best_match = None
                            best_similarity = 0
                            
                            employees = Employee.query.all()
                            for employee in employees:
                                employee_images = EmployeeImage.query.filter_by(employee_id=employee.id).all()
                                for emp_img in employee_images:
                                    if emp_img.arcface_embedding is not None:
                                        similarity = np.dot(embedding, emp_img.arcface_embedding)
                                        if similarity > best_similarity and similarity > FACE_RECOGNITION_THRESHOLD:
                                            best_similarity = similarity
                                            best_match = employee
                            
                            bbox = face.bbox.astype(int)
                            detection = {
                                'name': best_match.name if best_match else 'Unknown',
                                'confidence': float(best_similarity),
                                'bbox': bbox.tolist(),
                                'employee_id': best_match.employee_id if best_match else None
                            }
                            detections.append(detection)
                            
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                            label = f"{detection['name']} ({detection['confidence']:.2f})"
                            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if best_match and best_similarity > FACE_RECOGNITION_THRESHOLD:
                                cooldown_seconds = app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
                                recent_time = datetime.utcnow() - timedelta(seconds=cooldown_seconds)
                                
                                recent_log = AttendanceLog.query.filter(
                                    AttendanceLog.employee_id == best_match.id,
                                    AttendanceLog.camera_feed_name == camera_name,
                                    AttendanceLog.timestamp >= recent_time
                                ).first()
                                
                                if not recent_log:
                                    attendance_log = AttendanceLog(
                                        employee_name=best_match.name,
                                        employee_id=best_match.employee_id,
                                        camera_feed_name=camera_name,
                                        attendance_type='detected',
                                        confidence_score=float(best_similarity),
                                        timestamp=datetime.utcnow()
                                    )
                                    db.session.add(attendance_log)
                                    db.session.commit()
                                    
                                    print(f"[DETECTION] Employee detected: id={best_match.id}, office_employee_id={best_match.office_employee_id}, name={best_match.name}, confidence={best_similarity:.2f}")
                                    socketio.emit('detection_update', {
                                        'employee_id': best_match.id,
                                        'office_employee_id': best_match.office_employee_id,
                                        'employee_name': best_match.name,
                                        'camera_name': camera_name,
                                        'confidence': float(best_similarity),
                                        'timestamp': datetime.utcnow().strftime('%H:%M:%S')
                                    })
                
            except Exception as e:
                print(f"[ERROR] Face detection error for camera {camera_id}: {e}")
            
            time.sleep(0.033)
        
    except Exception as e:
        print(f"[ERROR] Camera processing error for {camera_id}: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        if camera_id in face_trackers:
            del face_trackers[camera_id]
        print(f"[INFO] Stopped processing camera {camera_id}")

@app.route('/system_info')
def system_info():
    now = datetime.utcnow()
    uptime_seconds = int((now - server_start_time).total_seconds())
    return jsonify({
        'ip': get_server_ip(),
        'uptime': uptime_seconds
    })

@app.route('/start_camera_processing/<int:camera_id>', methods=['POST'])
def start_camera_processing(camera_id):
    try:
        camera_feed = CameraFeed.query.get(camera_id)
        if not camera_feed:
            return jsonify({'status': 'error', 'message': 'Camera not found'}), 404
        
        if camera_id in camera_threads and camera_threads[camera_id].is_alive():
            return jsonify({'status': 'error', 'message': 'Camera already running'}), 400
        
        camera_active[camera_id] = True
        thread = Thread(target=process_camera_feed, args=(camera_id, camera_feed.name, camera_feed.camera_url))
        thread.daemon = True
        thread.start()
        camera_threads[camera_id] = thread
        
        return jsonify({'status': 'success', 'message': 'Camera processing started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_camera_processing/<int:camera_id>', methods=['POST'])
def stop_camera_processing(camera_id):
    try:
        camera_active[camera_id] = False
        if camera_id in camera_threads:
            camera_threads[camera_id].join(timeout=2)
            del camera_threads[camera_id]
        
        return jsonify({'status': 'success', 'message': 'Camera processing stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/camera_status')
def get_camera_status():
    try:
        active_cameras = []
        for camera_id, is_active in camera_active.items():
            if is_active and camera_id in camera_threads and camera_threads[camera_id].is_alive():
                camera_feed = CameraFeed.query.get(camera_id)
                if camera_feed:
                    active_cameras.append({
                        'id': camera_id,
                        'name': camera_feed.name,
                        'status': 'running'
                    })
        
        return jsonify({
            'active_cameras': active_cameras,
            'total_active': len(active_cameras)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera_feeds')
def get_camera_feeds():
    try:
        feeds = CameraFeed.query.all()
        result = []
        for feed in feeds:
            result.append({
                'id': feed.id,
                'name': feed.name,
                'camera_url': feed.camera_url,
                'camera_type': feed.camera_type,
                'location': feed.location,
                'description': feed.description,
                'is_active': feed.is_active,
                'created_at': feed.created_at.isoformat() if feed.created_at else None
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/settings_page')
def settings_page():
    server_ip = get_server_ip()
    ws_port = 5000
    webrtc_port = 5000
    mjpeg_port = 5000
    return render_template(
        'settings.html',
        server_ip=server_ip,
        ws_port=ws_port,
        webrtc_port=webrtc_port,
        mjpeg_port=mjpeg_port
    )

@app.route('/stream_viewer')
def stream_viewer():
    return render_template('stream_viewer.html')

# Debug: expose the absolute SQLite DB path being used by this process
@app.route('/api/_debug_db_path')
def debug_db_path():
    try:
        uri = app.config.get('SQLALCHEMY_DATABASE_URI')
        # Resolve common sqlite relative form like sqlite:///database.db
        db_file = None
        if uri and uri.startswith('sqlite:///'):
            relative = uri.replace('sqlite:///','',1)
            db_file = os.path.abspath(os.path.join(os.getcwd(), relative))
        return jsonify({
            'sqlalchemy_uri': uri,
            'resolved_sqlite_path': db_file
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/client')
def client_page():
    return render_template('client.html')

@app.route('/client.html')
def client_page_html():
    return render_template('client.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Start a single unified WebSocket server on 8765 (no Flask reloader duplication)
    start_ws_server()
    # Run without Flask reloader to avoid double-binding background threads
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)