import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;tcp|rtsp_flags;prefer_tcp|probesize;262144|analyzeduration;1000000|max_delay;5000000|buffer_size;1048576|loglevel;quiet'
)
os.environ.setdefault('ORT_LOG_SEVERITY_LEVEL', '4')
os.environ.setdefault('ORT_LOG_VERBOSITY_LEVEL', '0')
import logging
import atexit
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, current_app, send_from_directory, abort, has_request_context
from flask import cli as flask_cli
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event, func
from sqlalchemy.engine import Engine
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
import onnxruntime as ort
import warnings
try:
    from sqlalchemy.exc import SAWarning
    warnings.filterwarnings("ignore", category=SAWarning)
    try:
        from sqlalchemy.exc import LegacyAPIWarning
        warnings.filterwarnings("ignore", category=LegacyAPIWarning)
    except Exception:
        pass
except Exception:
    pass
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
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
logging.getLogger("websockets.client").setLevel(logging.CRITICAL)
logging.getLogger("app_folder.routes.detection_history").setLevel(logging.WARNING)

# Safely handle Colorama on Windows to avoid atexit reset_all errors
try:
    import colorama  # type: ignore
    # If Colorama was initialized elsewhere, deinit to remove its atexit hook
    try:
        colorama.deinit()
    except Exception:
        pass
    # Initialize minimal console fixes without wrapping streams aggressively
    try:
        colorama.just_fix_windows_console()
    except Exception:
        # Fallback to standard init but ignore stream issues
        try:
            colorama.init(convert=True, strip=False, autoreset=False)
        except Exception:
            pass

    def _safe_colorama_deinit():
        try:
            colorama.deinit()
        except Exception:
            pass

    atexit.register(_safe_colorama_deinit)
except Exception:
    pass


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Ensure SQLite enforces foreign key constraints."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass

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
                if str(camera_url).isdigit():
                    cap = cv2.VideoCapture(int(camera_url))
                else:
                    url = encode_rtsp_url(camera_url)
                    if "rtsp://" in url and "rtsp_transport" not in url:
                        sep = '&' if '?' in url else '?'
                        url += f"{sep}rtsp_transport=tcp"
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
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
            pts, time_base = await self.next_timestamp()
            video_frame.pts = pts
            video_frame.time_base = time_base
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("face_recognition_system")
logger.setLevel(logging.ERROR)

# Runtime logger for concise operational info (IP/ports, client connects, etc.)
runtime_logger = logging.getLogger("runtime")
runtime_logger.setLevel(logging.INFO)
if not runtime_logger.handlers:
    _rt_handler = logging.StreamHandler()
    _rt_handler.setLevel(logging.INFO)
    _rt_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    runtime_logger.addHandler(_rt_handler)
    runtime_logger.propagate = False

build_info = cv2.getBuildInformation()
ffmpeg_support = 'FFMPEG:' in build_info and 'YES' in build_info.split('FFMPEG:')[1].split('\n')[0]

try:
    # Silence noisy third-party startup chatter
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*LegacyAPIWarning.*")

    class _NullIO:
        def write(self, *_):
            pass
        def flush(self):
            pass

    import sys
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = _NullIO()
    sys.stderr = _NullIO()
    try:
        available_providers = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]
        arcface_app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection', 'recognition'],
            providers=providers
        )
        arcface_app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
    finally:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr
    logger.info("FaceAnalysis initialized")
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
app.config['UNKNOWN_SNAPSHOT_FOLDER'] = os.path.join('static', 'unknown')
app.config['DETECTION_COOLDOWN_SECONDS'] = 30  
app.config['POPUP_EMIT_COOLDOWN_SECONDS'] = 5  
try:
    # Force absolute URL building for background contexts
    app.config['SERVER_NAME'] = f"{get_server_ip()}:5000"
except Exception:
    pass

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UNKNOWN_SNAPSHOT_FOLDER'], exist_ok=True)

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

# Popup emission throttling (per-employee)
# Limit detection_event popup frequency to reduce duplicate popups
last_detection_emit_times = {}
last_detection_emit_lock = threading.Lock()

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
    
    # print(f"[CONNECT] Client connected: {client_info['ip']}, sid: {request.sid} (Total unique clients: {len(unique_clients)})")
    
    runtime_logger.info(f"Socket.IO connect {client_info['ip']} (sid={request.sid}) | unique={len(unique_clients)}")
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
    
    runtime_logger.info(f"Socket.IO disconnect {request.remote_addr} (sid={request.sid}) | unique={len(unique_clients)}")
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
    # Keep test_event silent in production; uncomment if needed
    # runtime_logger.info(f"test_event from {request.remote_addr} (sid={request.sid}) -> {data}")
    socketio.emit('test_response', {'message': 'Backend received your test event!'}, room=request.sid)
    return {'status': 'received'}

@socketio.on('test_detection_event')
def handle_test_detection_event():
    # runtime_logger.info(f"Manual test_detection_event triggered by {request.remote_addr} (sid={request.sid})")
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
    # print("[TEST_DETECTION] Emitting test detection_event with data:", test_detection)
    socketio.emit('detection_event', test_detection, room=sid)
    socketio.emit('detection_event', test_detection)
    # print("[TEST_DETECTION] Test detection_event emitted successfully to client and broadcast")
    return {'status': 'test_detection_emitted'}

@socketio.on('join_stream_viewer')
def handle_join_stream_viewer():
    global websocket_streaming_active
    # runtime_logger.info(f"join_stream_viewer by {request.remote_addr} (sid={request.sid})")
    stream_viewers.add(request.sid)
    viewer_count = len(stream_viewers)
    # print(f"[JOIN_STREAM_VIEWER] Client joined stream viewer: {request.remote_addr}, sid: {request.sid} (Total viewers: {viewer_count})")
    # print(f"[JOIN_STREAM_VIEWER] stream_viewers set now contains: {stream_viewers}")
    
    if not websocket_streaming_active:
        websocket_streaming_active = True
        # runtime_logger.info("WebSocket streaming activated (viewer count >=1)")
    else:
        # runtime_logger.info("WebSocket streaming already active")
        pass
    
    # print(f"[JOIN_STREAM_VIEWER] Final state - websocket_streaming_active: {websocket_streaming_active}, stream_viewers: {stream_viewers}")
    
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
                        designation='', department='', location=None, access=True, timestamp=None, image_url=None):
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
    
        
    try:
        if detections is not None and isinstance(detections, dict):
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
        
        now_str = datetime.utcnow().strftime('%H:%M:%S')
        timestamp = timestamp or now_str
        location = location or camera_name
        
        if employee_name is None or employee_name == '':
            employee_name = 'Unknown Visitor'
            access = False
            designation = ''
            department = ''
        
        # Throttle popup emission per employee (including Unknown Visitor)
        try:
            cooldown = app.config.get('POPUP_EMIT_COOLDOWN_SECONDS', 5)
        except Exception:
            cooldown = 5
        try:
            import time as _t
            throttle_key = (employee_name or 'Unknown Visitor').strip() or 'Unknown Visitor'
            with last_detection_emit_lock:
                last_t = last_detection_emit_times.get(throttle_key, 0.0)
                now_t = _t.time()
                if (now_t - last_t) < cooldown:
                    return
                last_detection_emit_times[throttle_key] = now_t
        except Exception:
            pass
        
        from app_folder.models.employee import Employee
        from flask import url_for
        resolved_image_url = None
        try:
            with app.app_context():
                # If an explicit image_url was provided, prefer it
                if image_url:
                    resolved_image_url = image_url
                else:
                    # Always build absolute URLs; SERVER_NAME is configured
                    external_flag = True
                    # Default avatar
                    resolved_image_url = url_for('static', filename='img/default-avatar.png', _external=external_flag)
                    if employee_name and employee_name != 'Unknown Visitor':
                        # Case/space-insensitive name match
                        emp_name_norm = (employee_name or '').strip().lower()
                        employee = Employee.query.filter(func.lower(Employee.name) == emp_name_norm).first()
                        if employee:
                            # Direct URL to the DB-backed first image
                            resolved_image_url = url_for('employee_image_by_id', employee_id=employee.id, _external=external_flag)
                            try:
                                # runtime_logger.info(f"[EMP_IMAGE] Known employee '{employee.name}' (id={employee.id}) image URL resolved from DB: {resolved_image_url}")
                                pass
                            except Exception:
                                pass
                        else:
                            # Fall back to by-name route as a URL
                            resolved_image_url = url_for('employee_image_by_name', employee_name=employee_name, _external=external_flag)
        except Exception as e:
            # Ensure image URL is never empty: hard-fallback to absolute default avatar
            try:
                with app.app_context():
                    resolved_image_url = url_for('static', filename='img/default-avatar.png', _external=True)
            except Exception:
                server = app.config.get('SERVER_NAME', '127.0.0.1:5000')
                resolved_image_url = f"http://{server}/static/img/default-avatar.png"
            try:
                # runtime_logger.info(f"[EMP_IMAGE][FALLBACK] Using default avatar due to exception: {e}")
                pass
            except Exception:
                pass

        detection = {
            'employee_name': employee_name,
            'confidence': round(float(confidence), 2),
            'designation': designation,
            'department': department,
            'location': location,
            'timestamp': timestamp,
            'access': bool(access),
            'camera_name': camera_name,
            'camera_id': camera_id,
            'image_url': resolved_image_url
        }
        # Log concise detection details to terminal for known employees
        try:
            if employee_name and employee_name != 'Unknown Visitor':
                # runtime_logger.info(f"[DETECTION] name='{employee_name}' conf={round(float(confidence), 2)} cam='{camera_name}' img='{resolved_image_url}'")
                pass
        except Exception:
            pass
        
        # print("[SOCKETIO] Emitting detection_event with fields:")
        # logger.info("[SOCKETIO] Emitting detection_event with fields:")
        # for k, v in detection.items():
        #     print(f"    {k}: {v}")
        #     logger.info(f"    {k}: {v}")
            
        try:
            socketio.emit('detection_event', detection)
            try:
                for viewer_sid in list(stream_viewers):
                    socketio.emit('detection_event', detection, room=viewer_sid)
            except Exception:
                pass
            # print("[SOCKETIO] detection_event emitted successfully.")
            logger.info("detection_event emitted successfully.")
            
            # print(f"[DEBUG] stream_viewers set contains: {stream_viewers}")
            if stream_viewers:
                # print(f"[SOCKETIO] Also emitting to {len(stream_viewers)} stream viewers")
                for viewer_sid in stream_viewers:
                    socketio.emit('detection_event', detection, room=viewer_sid)
            else:
                # print("[DEBUG] No stream viewers found, emitting globally only")
                pass
                
        except Exception as emit_exc:
            # print(f"[ERROR] Exception during socketio.emit: {emit_exc}")
            logger.error(f"Exception during socketio.emit: {emit_exc}")
        
        try:
            detection_broadcast_queue.put_nowait(detection)
        except Exception:
            pass
            
    except Exception as e:
        # print(f"[ERROR] Exception in emit_websocket_frame: {e}")
        logger.error(f"Exception in emit_websocket_frame: {e}")

db.init_app(app)
migrate = Migrate(app, db)

from app_folder.models.employee import Employee, EmployeeImage
from app_folder.models.camera_feed import CameraFeed

from app_folder.models.detection_history import DetectionHistory
try:
    from app_folder.models.attendance_log import AttendanceLog
except Exception:
    AttendanceLog = None
from app_folder.routes.detection_history import detection_history_bp

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.register_blueprint(detection_history_bp)

# Ensure SQLAlchemy sessions are cleaned up after each app context
@app.teardown_appcontext
def remove_db_session(exception=None):
    try:
        db.session.remove()
    except Exception:
        pass

# Attendance log deletion endpoints used by the frontend
@app.route('/delete_all_attendance_logs', methods=['DELETE'])
def delete_all_attendance_logs():
    """Delete all attendance logs. Returns JSON for the frontend."""
    try:
        if AttendanceLog is None:
            return jsonify({'status': 'success', 'message': 'AttendanceLog model not available; nothing to delete'}), 200
        deleted = AttendanceLog.query.delete(synchronize_session=False)
        db.session.commit()
        return jsonify({'status': 'success', 'deleted_count': deleted, 'message': 'All attendance logs deleted'}), 200
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_attendance_log/<int:attendance_id>', methods=['DELETE'])
def delete_attendance_log(attendance_id):
    """Delete a single attendance log by ID. Returns JSON."""
    try:
        if AttendanceLog is None:
            return jsonify({'status': 'error', 'message': 'AttendanceLog model not available'}), 404
        record = AttendanceLog.query.get(attendance_id)
        if record is None:
            return jsonify({'status': 'error', 'message': 'Record not found'}), 404
        db.session.delete(record)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Record deleted'}), 200
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500

def encode_rtsp_url(rtsp_url):
    if 'rtsp://' not in rtsp_url:
        return rtsp_url
    try:
        # Encode credentials if present
        prefix, rest = rtsp_url.split('://', 1)
        userinfo = ''
        host_and_path = rest
        if '@' in rest:
            creds, host_and_path = rest.rsplit('@', 1)
            if ':' in creds:
                user, pwd = creds.split(':', 1)
                user_enc = urllib.parse.quote(user, safe='')
                pwd_enc = urllib.parse.quote(pwd, safe='')
                userinfo = f"{user_enc}:{pwd_enc}@"
            else:
                userinfo = f"{urllib.parse.quote(creds, safe='')}@"

        # Parse query to append safe defaults
        if '?' in host_and_path:
            base, query = host_and_path.split('?', 1)
            q = dict(urllib.parse.parse_qsl(query, keep_blank_values=True))
        else:
            base, q = host_and_path, {}

        # Add robust flags if missing
        if 'rtsp_flags' not in q:
            q['rtsp_flags'] = 'prefer_tcp'
        if 'fflags' not in q:
            q['fflags'] = 'nobuffer'
        if 'stimeout' not in q:
            # 5s open/connect timeout in microseconds
            q['stimeout'] = '5000000'

        new_query = urllib.parse.urlencode(q)
        return f"{prefix}://{userinfo}{base}?{new_query}" if new_query else f"{prefix}://{userinfo}{base}"
    except Exception:
        return rtsp_url

@app.route('/employee_image/<int:employee_id>')
@app.route('/employeeimage/<int:employee_id>')
def employee_image_by_id(employee_id):
    """Serve the first employee image by employee ID using DB lookup.
    Falls back to default avatar if not available.
    """
    try:
        emp = Employee.query.get(employee_id)
        if not emp or not getattr(emp, 'images', None):
            try:
                # runtime_logger.info(f"[EMP_IMAGE][ROUTE] No images for employee_id={employee_id}; serving default avatar")
                pass
            except Exception:
                pass
            return redirect(url_for('static', filename='img/default-avatar.png'))
        filename = emp.images[0].image_filename
        # Resolve absolute upload directory under app root
        upload_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(upload_dir, filename)
        if not os.path.exists(file_path):
            try:
                # runtime_logger.info(f"[EMP_IMAGE][ROUTE] File missing: {file_path}; serving default avatar")
                pass
            except Exception:
                pass
            return redirect(url_for('static', filename='img/default-avatar.png'))
        try:
            # runtime_logger.info(f"[EMP_IMAGE][ROUTE] Serving file: {file_path}")
            pass
        except Exception:
            pass
        return send_from_directory(upload_dir, filename)
    except Exception:
        return redirect(url_for('static', filename='img/default-avatar.png'))

@app.route('/employee_image/by_name/<path:employee_name>')
def employee_image_by_name(employee_name):
    """Serve the first employee image by employee name using DB lookup.
    Useful when only name is available.
    """
    try:
        emp = Employee.query.filter_by(name=employee_name).first()
        if not emp:
            return redirect(url_for('static', filename='img/default-avatar.png'))
        return employee_image_by_id(emp.id)
    except Exception:
        return redirect(url_for('static', filename='img/default-avatar.png'))

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
        # Explicitly delete child images to avoid setting employee_id to NULL
        from app_folder.models.employee import EmployeeImage
        EmployeeImage.query.filter_by(employee_id=employee.id).delete(synchronize_session=False)
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

# Oval zone configuration for popup control (right-middle area)
OVAL_ZONE_CONFIG = {
    'enabled': True,
    'center_x_ratio': 0.75,  # 75% from left (right-middle)
    'center_y_ratio': 0.3,   # 50% from top (middle)
    'width_ratio': 0.25,     # 25% of frame width
    'height_ratio': 0.40,    # 40% of frame height
    'mask_enabled': True
}

# Easy placement/config for overlay elements (pixels)
# Adjust these offsets to reposition elements without touching code
DRAW_ELEMENTS_CONFIG = {
    'welcome_text': {'enabled': False, 'offset_x': 0, 'offset_y': 0},
    'desk': {'offset_x': +200, 'offset_y': 0},
    'arrow': {'offset_x': 0, 'offset_y': 0},
    'gate': {
        'offset_x': 0,       # +right / -left
        'offset_y': -40,     # +down / -up (default: move gate slightly up)
        'grill_offset_y': -80  # Moves grill area up significantly by default
    },
    # Desk label config
    'desk_label': {
        'enabled': True,
        'text': 'Desk',
        'left_padding': 0,     # positive to move right from centered position
        'font_scale': 0.7,
        'thickness': 2,
        'color': (230, 230, 230)
    },
    # Gate label config (text drawn on the gate body)
    'gate_label': {
        'enabled': True,
        'text': 'Gate',
        'left_padding': 0,      # positive to move right from centered position
        'font_scale': 0.7,
        'thickness': 2,
        'color': (230, 230, 240)
    },
    # Heading shown above the gate
    'gate_heading': {
        'enabled': True,
        'text': 'Welcome to IGDRONES',
        'offset_x': 0,
        'offset_y': -62,     # additional manual offset (negative moves further up)
        'bottom_padding': 18,  # space between gate roof and heading
        'bg_alpha': 0.35,
        'bg_color': (20, 20, 20),  # BGR
        'font_scale': 1.4,
        'thickness': 3
    }
}

def is_face_in_oval_zone(face_center_x, face_center_y, frame_width, frame_height):
    """
    Check if a face center point is within the oval zone.
    
    Args:
        face_center_x: X coordinate of face center
        face_center_y: Y coordinate of face center
        frame_width: Width of the video frame
        frame_height: Height of the video frame
    
    Returns:
        bool: True if face is within oval zone, False otherwise
    """
    if not OVAL_ZONE_CONFIG['enabled']:
        return True  # If oval zone is disabled, all faces trigger popups
    
    # Calculate oval zone parameters
    oval_center_x = frame_width * OVAL_ZONE_CONFIG['center_x_ratio']
    oval_center_y = frame_height * OVAL_ZONE_CONFIG['center_y_ratio']
    oval_width = frame_width * OVAL_ZONE_CONFIG['width_ratio']
    oval_height = frame_height * OVAL_ZONE_CONFIG['height_ratio']
    
    # Calculate semi-axes
    a = oval_width / 2  # horizontal semi-axis
    b = oval_height / 2  # vertical semi-axis
    
    # Check if point is inside ellipse using standard ellipse equation
    # ((x-h)²/a²) + ((y-k)²/b²) <= 1
    dx = face_center_x - oval_center_x
    dy = face_center_y - oval_center_y
    
    ellipse_value = (dx * dx) / (a * a) + (dy * dy) / (b * b)
    
    return ellipse_value <= 1.0

def draw_oval_zone(frame):
    """
    Draw the oval zone overlay on the frame.
    
    Args:
        frame: OpenCV frame to draw on
    
    Returns:
        frame: Frame with oval zone drawn
    """
    if not OVAL_ZONE_CONFIG['enabled']:
        return frame
    
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate oval zone parameters
    oval_center_x = int(frame_width * OVAL_ZONE_CONFIG['center_x_ratio'])
    oval_center_y = int(frame_height * OVAL_ZONE_CONFIG['center_y_ratio'])
    oval_width = int(frame_width * OVAL_ZONE_CONFIG['width_ratio'])
    oval_height = int(frame_height * OVAL_ZONE_CONFIG['height_ratio'])
    
    # Draw oval zone
    cv2.ellipse(frame, 
                (oval_center_x, oval_center_y),  # center
                (oval_width // 2, oval_height // 2),  # axes
                0,  # angle
                0, 360,  # start and end angles
                (0, 255, 0),  # green color
                2)  # thickness
    
    # Add zone label
    cv2.putText(frame, "ACCESS ZONE", 
                (oval_center_x - 60, oval_center_y - oval_height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def apply_oval_zone_mask(frame):
    """
    Black out everything outside the configured oval zone and draw the oval outline/label.
    """
    if not OVAL_ZONE_CONFIG['enabled'] or not OVAL_ZONE_CONFIG.get('mask_enabled', False):
        # Still draw the oval overlay if enabled but masking disabled
        return draw_oval_zone(frame)

    frame_h, frame_w = frame.shape[:2]
    # Oval parameters
    cx = int(frame_w * OVAL_ZONE_CONFIG['center_x_ratio'])
    cy = int(frame_h * OVAL_ZONE_CONFIG['center_y_ratio'])
    w = int(frame_w * OVAL_ZONE_CONFIG['width_ratio'])
    h = int(frame_h * OVAL_ZONE_CONFIG['height_ratio'])
    axes = (w // 2, h // 2)

    # Create mask with ellipse filled white on black background
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, thickness=-1)

    # Apply mask to keep only inside the oval
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # Draw oval outline and label on top for clarity
    cv2.ellipse(masked, (cx, cy), axes, 0, 0, 360, (0, 255, 0), 2)
    label_pos = (cx - axes[0], max(cy - axes[1] - 10, 10))
    cv2.putText(masked, "ACCESS ZONE", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Add bold welcome text at top-left corner with padding and soft background
    try:
        wt_cfg = DRAW_ELEMENTS_CONFIG.get('welcome_text', {})
        if wt_cfg.get('enabled', False):
            text = "Welcome to IGDRONES"
            pad_x = int(wt_cfg.get('padding_x', 76))
            pad_y = int(wt_cfg.get('padding_y', 46))
            off_x = int(wt_cfg.get('offset_x', 0))
            off_y = int(wt_cfg.get('offset_y', 0))
            bg_alpha = float(wt_cfg.get('bg_alpha', 0.35))  # 0..1
            bg_color = tuple(map(int, wt_cfg.get('bg_color', (20, 20, 20))))

            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1.1
            thickness = 3

            # Compute text size for background box
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x1 = pad_x + off_x
            y1 = pad_y + off_y
            # Baseline-adjusted origin for text (text draws from baseline)
            org = (x1, y1 + th)

            # Semi-transparent background behind text
            x2 = x1 + tw
            y2 = y1 + th + baseline
            x1c = max(0, x1 - 10); y1c = max(0, y1 - 10)
            x2c = min(frame_w - 1, x2 + 10); y2c = min(frame_h - 1, y2 + 8)
            overlay = masked.copy()
            cv2.rectangle(overlay, (x1c, y1c), (x2c, y2c), bg_color, -1)
            cv2.addWeighted(overlay, bg_alpha, masked, 1 - bg_alpha, 0, masked)

            # Strong outline then main text for boldness
            cv2.putText(masked, text, org, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(masked, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    except Exception:
        pass

    # Draw a simple desk illustration (left-bottom area) and an arrow pointing to the oval zone
    try:
        # Responsive desk sizing based on frame dimensions for a cartoonish look
        margin = 20
        desk_w = max(180, int(frame_w * 0.28))
        tabletop_h = max(22, int(frame_h * 0.035))
        leg_w = max(12, int(desk_w * 0.06))
        leg_h = max(50, int(frame_h * 0.08))
        # Position: left-bottom with margin
        dx = margin + int(DRAW_ELEMENTS_CONFIG.get('desk', {}).get('offset_x', 0))
        dy = frame_h - margin - leg_h - tabletop_h - 20 + int(DRAW_ELEMENTS_CONFIG.get('desk', {}).get('offset_y', 0))  # 20px space for label below

        # Tabletop
        cv2.rectangle(masked, (dx, dy), (dx + desk_w, dy + tabletop_h), (80, 160, 255), -1)  # light orange fill
        cv2.rectangle(masked, (dx, dy), (dx + desk_w, dy + tabletop_h), (30, 60, 100), 3)    # thicker outline

        # Legs
        cv2.rectangle(masked, (dx + int(desk_w * 0.08), dy + tabletop_h), (dx + int(desk_w * 0.08) + leg_w, dy + tabletop_h + leg_h), (30, 60, 100), -1)
        cv2.rectangle(masked, (dx + desk_w - int(desk_w * 0.14), dy + tabletop_h), (dx + desk_w - int(desk_w * 0.14) + leg_w, dy + tabletop_h + leg_h), (30, 60, 100), -1)

        # Monitor on desk
        mon_w = max(90, int(desk_w * 0.36))
        mon_h = max(32, int(tabletop_h * 1.6))
        mon_x = dx + int((desk_w - mon_w) * 0.35)
        mon_y = dy - mon_h - max(10, int(tabletop_h * 0.6))
        cv2.rectangle(masked, (mon_x, mon_y), (mon_x + mon_w, mon_y + mon_h), (40, 200, 255), -1)  # screen fill
        cv2.rectangle(masked, (mon_x, mon_y), (mon_x + mon_w, mon_y + mon_h), (0, 120, 200), 3)    # thicker bezel
        # Monitor stand
        stand_w = max(8, int(mon_w * 0.06))
        cv2.rectangle(masked, (mon_x + mon_w // 2 - stand_w // 2, dy - 6), (mon_x + mon_w // 2 + stand_w // 2, dy + 10), (0, 120, 200), -1)

        # Optional text label under desk (centered with left padding)
        desk_label_cfg = DRAW_ELEMENTS_CONFIG.get('desk_label', {})
        if desk_label_cfg.get('enabled', True):
            d_text = str(desk_label_cfg.get('text', 'Desk'))
            d_scale = float(desk_label_cfg.get('font_scale', 0.7))
            d_thick = int(desk_label_cfg.get('thickness', 2))
            d_lpad = int(desk_label_cfg.get('left_padding', 0))
            d_color = tuple(map(int, desk_label_cfg.get('color', (230, 230, 230))))
            (dtw, dth), dbase = cv2.getTextSize(d_text, cv2.FONT_HERSHEY_SIMPLEX, d_scale, d_thick)
            d_cx = dx + (desk_w - dtw) // 2 + d_lpad
            d_cy = dy + tabletop_h + leg_h + 18
            cv2.putText(masked, d_text, (d_cx, d_cy), cv2.FONT_HERSHEY_SIMPLEX, d_scale, d_color, d_thick, cv2.LINE_AA)

        # Arrow from desk toward the oval zone center
        start_pt = (dx + desk_w + 20 + int(DRAW_ELEMENTS_CONFIG.get('arrow', {}).get('offset_x', 0)),
                    dy + tabletop_h // 2 + int(DRAW_ELEMENTS_CONFIG.get('arrow', {}).get('offset_y', 0)))
        # Point near bottom-left boundary of the oval (slightly inset to keep visible)
        end_pt = (cx - axes[0] + 10, cy + axes[1] - 10)
        # Draw a thick outlined arrow for visibility
        cv2.arrowedLine(masked, start_pt, end_pt, (0, 0, 0), 10, tipLength=0.045)
        cv2.arrowedLine(masked, start_pt, end_pt, (0, 255, 255), 5, tipLength=0.045)  # yellow arrow
        # Add guidance text near the arrow start
        guide_y = max(30, dy - 10 + int(DRAW_ELEMENTS_CONFIG.get('arrow', {}).get('offset_y', 0)))
        cv2.putText(masked, "Proceed to Access Zone →", (start_pt[0] - 5, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass

    # Draw a gate with two pillars, roof, and grill; position customizable via DRAW_ELEMENTS_CONFIG
    try:
        g_margin_x = 200
        g_w = max(180, int(frame_w * 0.22))
        g_h = max(150, int(frame_h * 0.28))
        g_x = g_margin_x
        g_y = max(30, frame_h // 2 - g_h // 2)

        # Apply a 45° offset toward the top-right to approximate 4–6 cm visually
        # Using ~6% of the smaller frame dimension as a heuristic for the shift
        _offset = max(8, int(min(frame_w, frame_h) * 0.06))
        g_x = min(frame_w - g_w - 10, g_x + _offset)  # shift right, keep on screen
        g_y = max(10, g_y - _offset)                   # shift up, keep on screen

        # Apply user-configurable offsets (pixels)
        gate_cfg = DRAW_ELEMENTS_CONFIG.get('gate', {})
        g_x = min(frame_w - g_w - 10, max(10, g_x + int(gate_cfg.get('offset_x', 0))))
        g_y = min(frame_h - g_h - 10, max(10, g_y + int(gate_cfg.get('offset_y', 0))))

        pillar_w = max(16, int(g_w * 0.12))
        roof_h = max(24, int(g_h * 0.18))
        grill_gap = max(8, int(g_w * 0.05))

        # Pillars
        left_pillar_pts = (g_x, g_y + roof_h), (g_x + pillar_w, g_y + g_h)
        right_pillar_pts = (g_x + g_w - pillar_w, g_y + roof_h), (g_x + g_w, g_y + g_h)
        cv2.rectangle(masked, left_pillar_pts[0], left_pillar_pts[1], (70, 70, 90), -1)
        cv2.rectangle(masked, right_pillar_pts[0], right_pillar_pts[1], (70, 70, 90), -1)
        cv2.rectangle(masked, left_pillar_pts[0], left_pillar_pts[1], (180, 180, 200), 2)
        cv2.rectangle(masked, right_pillar_pts[0], right_pillar_pts[1], (180, 180, 200), 2)

        # Roof (simple triangular pediment)
        roof_base_left = (g_x, g_y + roof_h)
        roof_base_right = (g_x + g_w, g_y + roof_h)
        roof_peak = (g_x + g_w // 2, g_y)
        roof_pts = np.array([roof_base_left, roof_peak, roof_base_right], dtype=np.int32)
        cv2.fillPoly(masked, [roof_pts], (60, 120, 200))
        cv2.polylines(masked, [roof_pts], isClosed=True, color=(200, 230, 255), thickness=2)

        # Large heading centered above the gate
        try:
            gh = DRAW_ELEMENTS_CONFIG.get('gate_heading', {})
            head_text = str(gh.get('text', 'Welcome to IGDRONES'))
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = float(gh.get('font_scale', 1.4))
            thickness = int(gh.get('thickness', 3))
            off_x = int(gh.get('offset_x', 0))
            off_y = int(gh.get('offset_y', -12))
            bottom_pad = int(gh.get('bottom_padding', 18))
            bg_alpha = float(gh.get('bg_alpha', 0.35))
            bg_color = tuple(map(int, gh.get('bg_color', (20, 20, 20))))

            (tw, th), baseline = cv2.getTextSize(head_text, font, font_scale, thickness)
            center_x = g_x + g_w // 2
            # place heading above the roof with configurable bottom padding
            base_y = max(10, g_y - bottom_pad)
            x1 = center_x - tw // 2 + off_x
            y1 = base_y + off_y - th  # top-left of text box
            # Clamp box
            x1c = max(0, x1 - 12)
            y1c = max(0, y1 - 8)
            x2c = min(frame_w - 1, x1 + tw + 12)
            y2c = min(frame_h - 1, y1 + th + baseline + 8)
            overlay = masked.copy()
            cv2.rectangle(overlay, (x1c, y1c), (x2c, y2c), bg_color, -1)
            cv2.addWeighted(overlay, bg_alpha, masked, 1 - bg_alpha, 0, masked)
            # Draw text (baseline origin)
            org = (x1, y1 + th)
            cv2.putText(masked, head_text, org, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(masked, head_text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass

        # Grill area between pillars
        grill_left = g_x + pillar_w
        grill_right = g_x + g_w - pillar_w
        grill_top = g_y + roof_h + 6
        grill_bottom = g_y + g_h - 6
        # Move grill area up/down if configured
        grill_shift = int(DRAW_ELEMENTS_CONFIG.get('gate', {}).get('grill_offset_y', 0))
        grill_top = max(g_y + roof_h + 2, grill_top + grill_shift)
        grill_bottom = max(grill_top + 10, grill_bottom + grill_shift)
        # Outer frame
        cv2.rectangle(masked, (grill_left, grill_top), (grill_right, grill_bottom), (200, 200, 220), 2)
        # Vertical bars
        for vx in range(grill_left + grill_gap, grill_right, grill_gap):
            cv2.line(masked, (vx, grill_top + 4), (vx, grill_bottom - 4), (190, 190, 210), 2)
        # Horizontal bars
        for vy in [grill_top + int((grill_bottom - grill_top) * r) for r in (0.3, 0.6)]:
            cv2.line(masked, (grill_left + 3, vy), (grill_right - 3, vy), (190, 190, 210), 2)

        # Gate label (centered between pillars with optional left padding)
        gl = g_x + pillar_w
        gr = g_x + g_w - pillar_w
        label_cfg = DRAW_ELEMENTS_CONFIG.get('gate_label', {})
        if label_cfg.get('enabled', True):
            gate_text = str(label_cfg.get('text', 'Gate'))
            fscale = float(label_cfg.get('font_scale', 0.7))
            thick = int(label_cfg.get('thickness', 2))
            lpad = int(label_cfg.get('left_padding', 0))
            color = tuple(map(int, label_cfg.get('color', (230, 230, 240))))
            (gtw, gth), gbase = cv2.getTextSize(gate_text, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
            gx = gl + (gr - gl - gtw) // 2 + lpad
            gy = g_y + roof_h - 6
            cv2.putText(masked, gate_text, (gx, gy), cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick, cv2.LINE_AA)
    except Exception:
        pass

    return masked

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
                frame_h, frame_w = frame.shape[:2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if is_face_in_oval_zone(cx, cy, frame_w, frame_h):
                    color = (0, 0, 255) if name != "Unknown" else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except IndexError as ie:
                # print(f"[DRAW ERROR] IndexError for face #{i}: {ie}. Data: box={box}, name={name}")
                pass
            except Exception as draw_err:
                # print(f"[DRAW ERROR] General error for face #{i} ({name}): {draw_err}")
                pass
        
        # Apply oval mask to keep only the zone visible
        frame = apply_oval_zone_mask(frame)
        
        return frame
    except Exception as e:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Model error: {str(e)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

def gen_frames(video_filename=None, camera_feed_id=None):
    from flask import current_app
    with app.app_context():
        # print("[DEBUG] gen_frames called with video_filename=", video_filename, "camera_feed_id=", camera_feed_id)
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
                    # print(f"[DEBUG] Employee {e.name} has {len(img_embeddings)} embeddings.")
            else:
                # print(f"[DEBUG] No valid embeddings for employee {e.name}")
                pass
            # print(f"[DEBUG] Total employees: {len(employees)}, with embeddings: {len(known_arcface_embeddings)}")
        except Exception as e:
            # print(f"[DEBUG] Exception during employee DB query: {e}")
            error_frame = generate_error_frame(f"DB error: {str(e)}")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            # print("[DEBUG] Yielding DB error frame")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return

        camera = None
        frame_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        try:
            if camera_feed_id:
                # print(f"[DEBUG] Looking up CameraFeed with id {camera_feed_id}")
                camera_feed = CameraFeed.query.get(camera_feed_id)
                if not camera_feed or not camera_feed.is_active:
                    # print(f"[DEBUG] CameraFeed not found or not active: {camera_feed}")
                    error_frame = generate_error_frame("Camera feed not active")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    return
                if camera_feed.camera_type == 'device':
                    try:
                        cam_index = int(camera_feed.camera_url)
                        # print(f"[DEBUG] Attempting to open device camera index {cam_index}")
                        test_cam = cv2.VideoCapture(cam_index)
                        if not test_cam.isOpened():
                            # print(f"[DEBUG] Camera device index {cam_index} not available")
                            error_frame = generate_error_frame(f"Camera device index {cam_index} not available")
                            ret, buffer = cv2.imencode('.jpg', error_frame)
                            frame_bytes = buffer.tobytes()
                            # print("[DEBUG] Yielding device not available error frame")
                            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            test_cam.release()
                            return
                        test_cam.release()
                        camera = cv2.VideoCapture(cam_index)
                        # print(f"[DEBUG] Device camera {cam_index} opened: {camera.isOpened()}")
                    except ValueError:
                        # print(f"[DEBUG] Invalid camera index, trying to find available camera")
                        camera = find_available_camera()
                else:
                    rtsp_url = camera_feed.camera_url
                    rtsp_url = encode_rtsp_url(rtsp_url)
                    if "rtsp://" in rtsp_url and "rtsp_transport" not in rtsp_url:
                        sep = '&' if '?' in rtsp_url else '?'
                        rtsp_url += f"{sep}rtsp_transport=tcp"
                    # print(f"[DEBUG] Attempting to open RTSP stream: {rtsp_url}")
                    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    time.sleep(0.5)
                    retry_count = 0
                    max_retries = 5
                    while (camera is None or not camera.isOpened()) and retry_count < max_retries:
                        # print(f"[WARN] Failed to open RTSP stream. Retrying {retry_count+1}/{max_retries}...")
                        time.sleep(1.5)
                        camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        time.sleep(0.5)
                        retry_count += 1
                    # print(f"[DEBUG] RTSP camera opened: {camera.isOpened()}")
                    if camera is None or not camera.isOpened():
                        # print(f"[DEBUG] Camera not available (RTSP connect failed)")
                        error_frame = generate_error_frame("Camera not available (RTSP connect failed)")
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        frame_bytes = buffer.tobytes()
                        # print("[DEBUG] Yielding RTSP not available error frame")
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        return
            elif video_filename:
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                # print(f"[DEBUG] Attempting to open video file: {video_path}")
                camera = cv2.VideoCapture(video_path)
                # print(f"[DEBUG] Video file camera opened: {camera.isOpened()}")
            else:
                # print(f"[DEBUG] No camera_feed_id or video_filename, trying to find available camera")
                camera = find_available_camera()
                # print(f"[DEBUG] Available camera opened: {camera.isOpened() if camera else None}")

            if camera is None or not camera.isOpened():
                # print(f"[DEBUG] Camera not available at all")
                error_frame = generate_error_frame("Camera not available")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                # print("[DEBUG] Yielding camera not available error frame")
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return

            def camera_worker():
                # print("[DEBUG] camera_worker started")
                frame_count = 0
                start_time = time.time()
                fps = 0.0
                process_every_n = 5
                last_boxes = []
                last_names = []
                last_confidences = []
                last_emit_times = {}

                if not camera.isOpened():
                    # print("[ERROR] Camera stream could not be opened. Check RTSP URL, credentials, and permissions.")
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
                            try:
                                camera_name = camera_feed.name if camera_feed_id and 'camera_feed' in locals() and camera_feed else "Live Camera"
                            except Exception:
                                camera_name = "Live Camera"
                            from datetime import datetime
                            now_ts = time.time()
                            frame_height, frame_width = frame.shape[:2]
                            
                            # Check each face detection and only emit popup for faces in oval zone
                            for (x1, y1, x2, y2), name, confidence in zip(last_boxes, last_names, last_confidences):
                                # Calculate face center point
                                face_center_x = (x1 + x2) / 2
                                face_center_y = (y1 + y2) / 2
                                
                                # Check if face is in oval zone - only emit popup if it is
                                if is_face_in_oval_zone(face_center_x, face_center_y, frame_width, frame_height):
                                    key = name if name != "Unknown" else "Unknown"
                                    last_ts = last_emit_times.get(key, 0)
                                    if now_ts - last_ts < 1.0:
                                        continue
                                    timestamp = datetime.utcnow().strftime('%H:%M:%S')
                                    try:
                                        if name != "Unknown" and confidence > 0.6:
                                            employee_obj = next((e for e in employees if e.name == name), None)
                                            if employee_obj:
                                                # Resolve DB-backed image URL for this employee and pass it through
                                                try:
                                                    with app.app_context():
                                                        # Use relative URL (works without request context)
                                                        _img_url = url_for('employee_image_by_id', employee_id=employee_obj.id, _external=False)
                                                    # runtime_logger.info(f"[CALLSITE] Passing image_url for '{employee_obj.name}' -> {_img_url}")
                                                except Exception:
                                                    _img_url = None
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
                                                    timestamp=timestamp,
                                                    image_url=_img_url
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
                                            # Unknown person: capture a face snapshot and include its URL
                                            try:
                                                xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                                                # Clamp coordinates to frame bounds
                                                frame_h, frame_w = frame.shape[:2]
                                                xi1 = max(0, min(frame_w - 1, xi1))
                                                xi2 = max(0, min(frame_w - 1, xi2))
                                                yi1 = max(0, min(frame_h - 1, yi1))
                                                yi2 = max(0, min(frame_h - 1, yi2))
                                                _img_url = None
                                                if xi2 > xi1 and yi2 > yi1:
                                                    face_crop = frame[yi1:yi2, xi1:xi2]
                                                    if face_crop is not None and face_crop.size > 0:
                                                        # Prepare save path
                                                        ts_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                                                        fname = f"unknown_{ts_str}.jpg"
                                                        save_dir = os.path.join(app.root_path, app.config.get('UNKNOWN_SNAPSHOT_FOLDER', 'static/unknown'))
                                                        os.makedirs(save_dir, exist_ok=True)
                                                        save_path = os.path.join(save_dir, fname)
                                                        try:
                                                            cv2.imwrite(save_path, face_crop)
                                                        except Exception:
                                                            pass
                                                        # Build URL for frontend
                                                        try:
                                                            with app.app_context():
                                                                _img_url = url_for('static', filename=f"unknown/{fname}", _external=False)
                                                        except Exception:
                                                            _img_url = None
                                                emit_websocket_frame(
                                                    camera_id=camera_feed_id or 0,
                                                    camera_name=camera_name,
                                                    frame=frame.copy(),
                                                    employee_name="Unknown Visitor",
                                                    confidence=confidence,
                                                    access=False,
                                                    timestamp=timestamp,
                                                    image_url=_img_url
                                                )
                                            except Exception:
                                                # Fallback emit without image if snapshot fails
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
                                        # print(f"[EMIT ERROR] Could not emit detection event: {emit_err}")
                                        pass

                        for (x1, y1, x2, y2), name, confidence in zip(last_boxes, last_names, last_confidences):
                            frame_h, frame_w = frame.shape[:2]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            if is_face_in_oval_zone(cx, cy, frame_w, frame_h):
                                color = (255, 0, 0) if name != "Unknown" else (0, 255, 255)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        # Apply oval mask to keep only the zone visible
                        frame = apply_oval_zone_mask(frame)

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
                        # print(f"[ERROR] Exception in camera_worker: {e}")
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
                    # print(f"[DEBUG] Frame queue empty, yielding waiting frame")
                    error_frame = generate_error_frame("Waiting for camera...")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except GeneratorExit:
                    # Client disconnected; break cleanly
                    break
                except Exception as e:
                    # print(f"[ERROR] Exception in gen_frames yield loop: {e}")
                    error_frame = generate_error_frame(f"Yield error: {str(e)}")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    break
        except Exception as e:
            # print(f"[ERROR] Error in gen_frames: {e}")
            error_frame = generate_error_frame(f"Error: {str(e)}")
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            # print("[DEBUG] Yielding top-level error frame")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            stop_event.set()
            if camera is not None:
                camera.release()
            # print("[DEBUG] gen_frames finished, camera released")

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
            # print(f"Error trying camera {i}: {e}")
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
            jpg_data = extract_jpeg_bytes(frame_bytes)
            if jpg_data:
                try:
                    jpg_as_text = base64.b64encode(jpg_data).decode('utf-8')
                    message = json.dumps({"type": "video_frame", "frame": jpg_as_text})
                    await websocket.send(message)
                except Exception:
                    pass

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
        try:
            async with websockets.serve(ws_video_stream, "0.0.0.0", 8765):
                # Single consolidated line printed at startup; avoid duplicate here
                pass
                await asyncio.Future()
        except OSError as e:
            if e.errno == 10048:  # Address already in use
                # Avoid extra port-in-use chatter; we already summarize at startup
                pass
                try:
                    async with websockets.serve(ws_video_stream, "0.0.0.0", 8766):
                        # Avoid duplicate log
                        pass
                        await asyncio.Future()
                except OSError as e2:
                    runtime_logger.error(f"Failed to start WebSocket server on port 8766: {e2}")
                    pass
            else:
                runtime_logger.error(f"Failed to start WebSocket server: {e}")
                pass
    def run_ws():
        try:
            asyncio.run(ws_main())
        except Exception as e:
            runtime_logger.error(f"WebSocket server failed to start: {e}")
            pass
    t = threading.Thread(target=run_ws, daemon=True)
    t.start()

detection_broadcast_queue = queue.Queue()

@app.route('/live_detection')
def live_detection():
    from flask import Response, stream_with_context
    video_filename = request.args.get('video')
    camera_feed_id = request.args.get('camera_feed_id')
    if camera_feed_id:
        camera_feed_id = int(camera_feed_id)
    return Response(stream_with_context(gen_frames(video_filename, camera_feed_id)), mimetype='multipart/x-mixed-replace; boundary=frame')

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
            if "rtsp://" in rtsp_url and "rtsp_transport" not in rtsp_url:
                sep = '&' if '?' in rtsp_url else '?'
                rtsp_url += f"{sep}rtsp_transport=tcp"
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
            rtsp_url = encode_rtsp_url(camera_feed.camera_url)
            if "rtsp://" in rtsp_url and "rtsp_transport" not in rtsp_url:
                sep = '&' if '?' in rtsp_url else '?'
                rtsp_url += f"{sep}rtsp_transport=tcp"
            camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

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
            url = encode_rtsp_url(camera_url)
            if "rtsp://" in url and "rtsp_transport" not in url:
                sep = '&' if '?' in url else '?'
                url += f"{sep}rtsp_transport=tcp"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            # print(f"[ERROR] Cannot open camera {camera_id}: {camera_url}")
            return
        
        face_trackers[camera_id] = FaceTracker()
        
        # print(f"[INFO] Started processing camera {camera_id}: {camera_name}")
        
        while camera_active.get(camera_id, False):
            ret, frame = cap.read()
            if not ret:
                # print(f"[WARN] Failed to read frame from camera {camera_id}")
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
                            
                            frame_h, frame_w = frame.shape[:2]
                            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                            if is_face_in_oval_zone(cx, cy, frame_w, frame_h):
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                                label = f"{detection['name']} ({detection['confidence']:.2f})"
                                cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if best_match and best_similarity > FACE_RECOGNITION_THRESHOLD:
                                cooldown_seconds = app.config.get('DETECTION_COOLDOWN_SECONDS', 30)
                                recent_time = datetime.utcnow() - timedelta(seconds=cooldown_seconds)
                                recent_log = None
                                try:
                                    if AttendanceLog is not None:
                                        with app.app_context():
                                            recent_log = AttendanceLog.query.filter(
                                                AttendanceLog.employee_id == best_match.id,
                                                AttendanceLog.camera_feed_name == camera_name,
                                                AttendanceLog.timestamp >= recent_time
                                            ).first()
                                except Exception:
                                    recent_log = None
                                finally:
                                    try:
                                        db.session.remove()
                                    except Exception:
                                        pass

                                if not recent_log and AttendanceLog is not None:
                                    try:
                                        with app.app_context():
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
                                    except Exception:
                                        try:
                                            db.session.rollback()
                                        except Exception:
                                            pass
                                    finally:
                                        try:
                                            db.session.remove()
                                        except Exception:
                                            pass
                                    
                                    # print(f"[DETECTION] Employee detected: id={best_match.id}, office_employee_id={best_match.office_employee_id}, name={best_match.name}, confidence={best_similarity:.2f}")
                                    socketio.emit('detection_update', {
                                        'employee_id': best_match.id,
                                        'office_employee_id': best_match.office_employee_id,
                                        'employee_name': best_match.name,
                                        'camera_name': camera_name,
                                        'confidence': float(best_similarity),
                                        'timestamp': datetime.utcnow().strftime('%H:%M:%S')
                                    })
                
            except Exception as e:
                # print(f"[ERROR] Face detection error for camera {camera_id}: {e}")
                pass
            
            time.sleep(0.033)
        
    except Exception as e:
        # print(f"[ERROR] Camera processing error for {camera_id}: {e}")
        pass
    finally:
        if 'cap' in locals():
            cap.release()
        if camera_id in face_trackers:
            del face_trackers[camera_id]
        # print(f"[INFO] Stopped processing camera {camera_id}")

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

@app.route('/api/_debug_db_path')
def debug_db_path():
    try:
        uri = app.config.get('SQLALCHEMY_DATABASE_URI')
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

@app.route('/api/_debug_employees')
def debug_employees():
    try:
        employees = Employee.query.all()
        employee_data = []
        for emp in employees:
            emp_info = {
                'id': emp.id,
                'name': emp.name,
                'office_employee_id': emp.office_employee_id,
                'images_count': len(emp.images),
                'images': []
            }
            for img in emp.images:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.image_filename)
                emp_info['images'].append({
                    'filename': img.image_filename,
                    'full_path': img_path,
                    'exists': os.path.exists(img_path)
                })
            employee_data.append(emp_info)
        
        return jsonify({
            'employees': employee_data,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            'upload_folder_contents': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fix_missing_images', methods=['POST'])
def fix_missing_images():
    """Fix missing employee images by creating placeholder images"""
    try:
        employees = Employee.query.all()
        fixed_count = 0
        
        for emp in employees:
            for img in emp.images:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.image_filename)
                if not os.path.exists(img_path):
                    # Create a simple placeholder image
                    import numpy as np
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    placeholder[:] = (100, 100, 100)  # Gray background
                    
                    # Add text
                    import cv2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(placeholder, emp.name[:10], (10, 100), font, 1, (255, 255, 255), 2)
                    cv2.putText(placeholder, "Missing", (10, 150), font, 0.7, (255, 255, 255), 2)
                    
                    # Save the placeholder
                    cv2.imwrite(img_path, placeholder)
                    fixed_count += 1
            # print(f"[FIX] Created placeholder for {emp.name}: {img.image_filename}")
        
        return jsonify({
            'success': True,
            'message': f'Created {fixed_count} placeholder images for missing files',
            'fixed_count': fixed_count
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
    start_ws_server()
    server_host = '0.0.0.0'
    server_port = 5000
    runtime_logger.info(f"Server: http://{get_server_ip()}:{server_port} | WS: ws://{get_server_ip()}:8765 (fallback :8766)")
    socketio.run(app, host=server_host, port=server_port, debug=False, use_reloader=False)