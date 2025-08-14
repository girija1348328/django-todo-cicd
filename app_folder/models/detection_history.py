"""
DetectionHistory model for storing face detection and recognition events.
"""
from app_folder.extensions import db
from datetime import datetime

class DetectionHistory(db.Model):
    __tablename__ = 'detection_history'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Detection result
    access_granted = db.Column(db.Boolean, nullable=False)  # True for granted, False for denied
    
    # Person information
    employee_name = db.Column(db.String(100), nullable=True)  # Could be "Unknown Visitor" for denied
    designation = db.Column(db.String(100), nullable=True)
    department = db.Column(db.String(100), nullable=True)
    
    # Detection details
    confidence_score = db.Column(db.Float, nullable=True)
    location = db.Column(db.String(100), nullable=True)  # Camera location/name
    
    # Camera information
    camera_name = db.Column(db.String(100), nullable=True)
    camera_id = db.Column(db.String(50), nullable=True)
    
    # Additional metadata
    detection_type = db.Column(db.String(50), nullable=True)  # 'face_detection', 'face_recognition', etc.
    image_data = db.Column(db.Text, nullable=True)  # Base64 encoded image if needed
    
    # Employee reference (if known) - removed foreign key constraint
    employee_id = db.Column(db.Integer, nullable=True)
    
    def __repr__(self):
        status = "GRANTED" if self.access_granted else "DENIED"
        return f'<DetectionHistory {self.employee_name} - {status} at {self.timestamp}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'access_granted': self.access_granted,
            'employee_name': self.employee_name,
            'designation': self.designation,
            'department': self.department,
            'confidence_score': self.confidence_score,
            'location': self.location,
            'camera_name': self.camera_name,
            'camera_id': self.camera_id,
            'detection_type': self.detection_type,
            'employee_id': self.employee_id
        }
