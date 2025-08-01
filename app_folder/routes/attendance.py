"""
Attendance-related routes for the Criminal Face Detection system.
"""
from flask import Blueprint, request, jsonify
from ..models.attendance_log import AttendanceLog
from ..extensions import db

attendance_bp = Blueprint('attendance', __name__)

@attendance_bp.route('/attendance_logs', methods=['GET'])
def attendance_logs():
    """
    List all attendance logs.
    """
    logs = AttendanceLog.query.all()
    result = []
    for log in logs:
        result.append({
            'id': log.id,
            'timestamp': log.timestamp.isoformat() if log.timestamp else None,
            'employee_name': log.employee_name,
            'attendance_type': log.attendance_type,
            'camera_feed_id': log.camera_feed_id,
            'camera_feed_name': log.camera_feed_name,
            'confidence_score': log.confidence_score
        })
    return jsonify(result) 