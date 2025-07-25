"""
CameraFeed model for the Criminal Face Detection system.
"""
from ..extensions import db
from datetime import datetime

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