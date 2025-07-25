"""
Employee model for the Criminal Face Detection system.
"""
from ..extensions import db

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    image_filename = db.Column(db.String(200), nullable=False)
    face_encoding = db.Column(db.PickleType, nullable=False)  # Store numpy array as binary

    def __repr__(self):
        return f'<Employee {self.name}>' 