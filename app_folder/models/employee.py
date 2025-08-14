"""
Employee model for the Criminal Face Detection system.
"""
from ..extensions import db

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    office_employee_id = db.Column(db.String(50), unique=True, nullable=False)  # Custom office-specific ID
    name = db.Column(db.String(100), nullable=False)
    designation = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    # Do NOT define images = db.relationship(...) here

    def __repr__(self):
        return f'<Employee {self.office_employee_id}, {self.name}, {self.designation}, {self.department}, {self.location}>'

class EmployeeImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    image_filename = db.Column(db.String(200), nullable=False)
    arcface_embedding = db.Column(db.PickleType, nullable=True)  # Store numpy array as binary

# Set the relationship using the class, not a string
# Ensure a deterministic order: first image = lowest id (earliest added)
Employee.images = db.relationship(
    EmployeeImage,
    backref='employee',
    lazy=True,
    order_by=EmployeeImage.id.asc()
)