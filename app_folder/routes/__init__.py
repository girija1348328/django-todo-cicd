"""
Blueprint registration for Flask app.
"""
from .employees import employees_bp
from .cameras import cameras_bp
from .attendance import attendance_bp

def register_blueprints(app):
    """
    Register all blueprints with the Flask app.
    """
    app.register_blueprint(employees_bp)
    app.register_blueprint(cameras_bp)
    app.register_blueprint(attendance_bp) 