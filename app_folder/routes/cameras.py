"""
Camera-related routes for the Criminal Face Detection system.
"""
from flask import Blueprint, request, jsonify, current_app as app
from ..models.camera_feed import CameraFeed
from ..extensions import db

cameras_bp = Blueprint('cameras', __name__)

@cameras_bp.route('/add_camera_feed', methods=['POST'])
def add_camera_feed():
    """
    Add a new camera feed.
    """
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@cameras_bp.route('/camera_feeds', methods=['GET'])
def list_camera_feeds():
    """
    List all camera feeds.
    """
    feeds = CameraFeed.query.all()
    result = []
    for f in feeds:
        result.append({
            'id': f.id,
            'name': f.name,
            'camera_url': f.camera_url,
            'camera_type': f.camera_type,
            'location': f.location,
            'description': f.description,
            'is_active': f.is_active,
            'created_at': f.created_at.isoformat() if f.created_at else None
        })
    return jsonify(result) 