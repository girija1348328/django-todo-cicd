"""
Attendance-related routes for the Criminal Face Detection system.
"""
from flask import Blueprint, request, jsonify

from ..extensions import db

attendance_bp = Blueprint('attendance', __name__)