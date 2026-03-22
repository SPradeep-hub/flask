# app/__init__.py
import os
from flask import Flask

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, '..', 'uploads')   # one level up from app/
    app.config['FRAMES_FOLDER'] = os.path.join(app.root_path, '..', 'frames')
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # 200 MB

    # Ensure the folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)

    # Register blueprint
    from .routes import main
    app.register_blueprint(main)

    return app