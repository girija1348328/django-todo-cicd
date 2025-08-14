"""
Routes for handling detection history operations.
"""
from flask import Blueprint, request, jsonify
from app_folder.extensions import db
from app_folder.models.detection_history import DetectionHistory
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detection_history_bp = Blueprint('detection_history', __name__)

@detection_history_bp.route('/api/detection_history', methods=['POST'])
def store_detection():
    """Store a new detection event in the database"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract data from the detection event
        access_granted = data.get('access', False)
        employee_name = data.get('employee_name', 'Unknown Visitor')
        designation = data.get('designation', 'N/A')
        department = data.get('department', 'N/A')
        confidence_score = data.get('confidence')
        location = data.get('location', 'Unknown Location')
        camera_name = data.get('camera_name', 'Unknown Camera')
        camera_id = data.get('camera_id', 'Unknown')
        detection_type = data.get('detection_type', 'face_detection')
        
        # Create new detection history record
        detection = DetectionHistory(
            access_granted=access_granted,
            employee_name=employee_name,
            designation=designation,
            department=department,
            confidence_score=confidence_score,
            location=location,
            camera_name=camera_name,
            camera_id=camera_id,
            detection_type=detection_type,
            employee_id=None  # Set to None since Employee model doesn't exist
        )
        
        db.session.add(detection)
        db.session.commit()
        
        logger.info(f"Stored detection: {employee_name} - {'GRANTED' if access_granted else 'DENIED'} at {location}")
        
        return jsonify({
            'success': True,
            'message': 'Detection stored successfully',
            'detection_id': detection.id
        }), 201
        
    except Exception as e:
        logger.error(f"Error storing detection: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to store detection: {str(e)}'}), 500

@detection_history_bp.route('/api/detection_history', methods=['GET'])
def get_detection_history():
    """Retrieve detection history with optional filtering"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        access_filter = request.args.get('access', type=str)  # 'granted', 'denied', or None for all
        camera_filter = request.args.get('camera', type=str)
        date_from = request.args.get('date_from', type=str)
        date_to = request.args.get('date_to', type=str)
        
        # Build query
        query = DetectionHistory.query
        
        # Apply filters
        if access_filter:
            if access_filter.lower() == 'granted':
                query = query.filter(DetectionHistory.access_granted == True)
            elif access_filter.lower() == 'denied':
                query = query.filter(DetectionHistory.access_granted == False)
        
        if camera_filter:
            query = query.filter(DetectionHistory.camera_name.contains(camera_filter))
        
        if date_from:
            try:
                date_from_obj = datetime.fromisoformat(date_from)
                query = query.filter(DetectionHistory.timestamp >= date_from_obj)
            except ValueError:
                pass
        
        if date_to:
            try:
                date_to_obj = datetime.fromisoformat(date_to)
                query = query.filter(DetectionHistory.timestamp <= date_to_obj)
            except ValueError:
                pass
        
        # Order by timestamp (newest first) and apply pagination
        query = query.order_by(DetectionHistory.timestamp.desc())
        total_count = query.count()
        detections = query.offset(offset).limit(limit).all()
        
        # Convert to list of dictionaries
        detection_list = [detection.to_dict() for detection in detections]
        
        return jsonify({
            'success': True,
            'detections': detection_list,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving detection history: {str(e)}")
        return jsonify({'error': f'Failed to retrieve detection history: {str(e)}'}), 500

@detection_history_bp.route('/api/detection_history/stats', methods=['GET'])
def get_detection_stats():
    """Get statistics about detection history"""
    try:
        # Get date range from query parameters
        days = request.args.get('days', 7, type=int)
        date_from = datetime.utcnow() - timedelta(days=days)
        
        # Get counts
        total_detections = DetectionHistory.query.filter(
            DetectionHistory.timestamp >= date_from
        ).count()
        
        granted_count = DetectionHistory.query.filter(
            DetectionHistory.timestamp >= date_from,
            DetectionHistory.access_granted == True
        ).count()
        
        denied_count = DetectionHistory.query.filter(
            DetectionHistory.timestamp >= date_from,
            DetectionHistory.access_granted == False
        ).count()
        
        # Get camera breakdown
        camera_stats = db.session.query(
            DetectionHistory.camera_name,
            db.func.count(DetectionHistory.id).label('count')
        ).filter(
            DetectionHistory.timestamp >= date_from
        ).group_by(DetectionHistory.camera_name).all()
        
        camera_breakdown = {stat.camera_name or 'Unknown': stat.count for stat in camera_stats}
        
        return jsonify({
            'success': True,
            'stats': {
                'total_detections': total_detections,
                'granted_count': granted_count,
                'denied_count': denied_count,
                'granted_percentage': round((granted_count / total_detections * 100) if total_detections > 0 else 0, 2),
                'denied_percentage': round((denied_count / total_detections * 100) if total_detections > 0 else 0, 2),
                'camera_breakdown': camera_breakdown,
                'date_range_days': days
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving detection stats: {str(e)}")
        return jsonify({'error': f'Failed to retrieve detection stats: {str(e)}'}), 500

@detection_history_bp.route('/api/detection_history/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    """Delete a specific detection record"""
    try:
        detection = DetectionHistory.query.get_or_404(detection_id)
        db.session.delete(detection)
        db.session.commit()
        
        logger.info(f"Deleted detection record ID: {detection_id}")
        
        return jsonify({
            'success': True,
            'message': 'Detection record deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting detection record: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to delete detection record: {str(e)}'}), 500

@detection_history_bp.route('/api/detection_history/clear', methods=['POST'])
def clear_detection_history():
    """Clear all detection history (use with caution)"""
    try:
        # Get confirmation parameter
        confirm = request.args.get('confirm', 'false')
        if confirm.lower() != 'true':
            return jsonify({'error': 'Confirmation required. Add ?confirm=true to confirm deletion.'}), 400
        
        # Delete all records
        deleted_count = DetectionHistory.query.delete()
        db.session.commit()
        
        logger.warning(f"Cleared all detection history. Deleted {deleted_count} records.")
        
        return jsonify({
            'success': True,
            'message': f'Cleared all detection history. Deleted {deleted_count} records.',
            'deleted_count': deleted_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing detection history: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to clear detection history: {str(e)}'}), 500
