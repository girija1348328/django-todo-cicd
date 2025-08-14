"""
Model registration for Flask app context.
"""
from .camera_feed import CameraFeed
from .detection_history import DetectionHistory

def register_models(app):
    """
    Register models for Flask shell context and migrations.
    """
    with app.app_context():
        CameraFeed.register(app)
        DetectionHistory.register(app)
    pass 