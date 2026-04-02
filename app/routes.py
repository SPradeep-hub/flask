from flask import Blueprint, render_template, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename

main = Blueprint("main", __name__)

# Home route
@main.route("/")
def home():
    return render_template("index.html")

# Upload route
@main.route('/upload_video', methods=['POST'])
def upload_video():

    print("UPLOAD ROUTE HIT")

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    file.save(temp_path)
    print("Saved at:", temp_path)

    # Run your script
    script_path = os.path.join(current_app.root_path, "processing", "00-convert_video_to_image.py")
    os.system(f"python \"{script_path}\" \"{temp_path}\"")
    
    return jsonify({
        'success': True,
        'message': 'Video uploaded and processing started'
    })