"""
AttendanceLog model for the Criminal Face Detection system.
"""
from ..extensions import db
from datetime import datetime

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