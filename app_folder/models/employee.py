"""
Employee model for the Criminal Face Detection system.
"""
from ..extensions import db

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    office_employee_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    designation = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<Employee {self.office_employee_id}, {self.name}, {self.designation}, {self.department}, {self.location}>'

class EmployeeImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(
        db.Integer,
        db.ForeignKey('employee.id', ondelete='CASCADE'),
        nullable=False
    )
    image_filename = db.Column(db.String(200), nullable=False)
    arcface_embedding = db.Column(db.PickleType, nullable=True)

Employee.images = db.relationship(
    EmployeeImage,
    backref=db.backref('employee'),
    lazy=True,
    order_by=EmployeeImage.id.asc(),
    cascade="all, delete, delete-orphan"
)