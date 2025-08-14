"""
Blueprint registration for Flask app.
"""
from .cameras import cameras_bp
from .attendance import attendance_bp
from .detection_history import detection_history_bp

def register_blueprints(app):
    """
    Register all blueprints with the Flask app.
    """
    app.register_blueprint(cameras_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(detection_history_bp) 