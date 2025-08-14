import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

def create_standalone_app():
    """Create a minimal Flask app for DB operations without starting the main app."""
    app = Flask(__name__)
    instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
    db_path = os.path.join(instance_path, 'database.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
        print(f'Created instance directory at {instance_path}')
        
    return app, db_path

app, DB_PATH = create_standalone_app()
db = SQLAlchemy(app)

from app_folder.models.employee import Employee, EmployeeImage
from app_folder.models.camera_feed import CameraFeed


def reset_database(force=False):
    """Deletes and recreates the database file and schema."""
    if not force:
        confirm = input(f'This will DELETE ALL data in {DB_PATH}. Type "yes" to continue: ')
        if confirm.strip().lower() != 'yes':
            print('Aborted.')
            return

    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f'Deleted existing database at {DB_PATH}')
        except Exception as e:
            print(f'Error deleting database: {e}')
            sys.exit(1)

    upload_folder = os.path.join(os.path.dirname(DB_PATH), 'uploads')
    if os.path.exists(upload_folder):
        try:
            import shutil
            shutil.rmtree(upload_folder)
            print(f'Deleted upload folder at {upload_folder}')
        except Exception as e:
            print(f'Error deleting upload folder: {e}')
            sys.exit(1)
    else:
        print('No existing database found. A new one will be created.')

    with app.app_context():
        print('Creating new database schema...')
        db.create_all()
        print('Database has been reset successfully.')

if __name__ == '__main__':
    force_reset = '--force' in sys.argv
    reset_database(force=force_reset)
