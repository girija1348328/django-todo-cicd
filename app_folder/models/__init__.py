"""
Model registration for Flask app context.
"""
from .employee import Employee
from .camera_feed import CameraFeed
from .attendance_log import AttendanceLog

def register_models(app):
    """
    Register models for Flask shell context and migrations.
    """
    pass 