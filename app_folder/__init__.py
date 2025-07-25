"""
App factory for the Criminal Face Detection Flask application.
"""
from flask import Flask
from .config import config
from .extensions import db
from .models import register_models
from .routes import register_blueprints

def create_app(config_name='default'):
    """
    Application factory for creating Flask app instances.
    :param config_name: The configuration to use (default, development, production, testing)
    :return: Configured Flask app
    """
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    db.init_app(app)
    register_models(app)
    register_blueprints(app)
    return app 