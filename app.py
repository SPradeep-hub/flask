import os
import subprocess
import sys
import uuid
import zipfile
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# ---------- Flask app initialization ----------
app = Flask(__name__,
            template_folder="app/templates",
            static_folder="app/static")

# ---------- Path configuration (must be before routes) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
FRAMES_BASE = os.path.join(BASE_DIR, 'frames')      # where frame images are stored
FACES_BASE = os.path.join(BASE_DIR, 'faces')        # where cropped faces will be stored

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_BASE, exist_ok=True)
os.makedirs(FACES_BASE, exist_ok=True)

# ---------- Video upload & processing endpoint ----------
@app.route("/upload_video", methods=['POST'])
def upload_video():
    print("=== UPLOAD ENDPOINT HIT ===")
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    # Validate extension
    allowed = {'mp4', 'mov', 'webm', 'avi', 'mkv'}
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        return jsonify({'success': False, 'error': f'Invalid file type: {ext}'}), 400

    # Save video with unique name
    original_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    safe_filename = f"{unique_id}.{ext}"
    video_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(video_path)
    print(f"✅ Video saved to: {video_path}")

    video_name = os.path.splitext(safe_filename)[0]

    # ---- Step 2: Extract frames using your existing script ----
    frame_extractor_script = os.path.join(BASE_DIR, 'processing', '00-convert_video_to_image.py')
    if not os.path.exists(frame_extractor_script):
        return jsonify({'success': False, 'error': 'Frame extractor script not found'}), 500

    frames_output_dir = os.path.join(FRAMES_BASE, video_name)
    os.makedirs(frames_output_dir, exist_ok=True)

    try:
        # Pass video path and output directory to the script
        result = subprocess.run(
            [sys.executable, frame_extractor_script, video_path, frames_output_dir],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            return jsonify({'success': False, 'error': f'Frame extraction failed: {result.stderr}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    # ---- Step 3: Crop faces from the extracted frames ----
    face_cropper_script = os.path.join(BASE_DIR, 'processing', '01b-crop_faces_from_frames.py')
    if not os.path.exists(face_cropper_script):
        return jsonify({'success': False, 'error': 'Face cropper script not found'}), 500

    faces_output_dir = os.path.join(FACES_BASE, video_name)
    try:
        result = subprocess.run(
            [sys.executable, face_cropper_script, frames_output_dir, faces_output_dir, 'dnn'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            # Face detection may fail if no faces – not a critical error
            print(f"Face cropping warning: {result.stderr}")
    except Exception as e:
        print(f"Face cropping error: {e}")

    # ---- Step 4: Optionally create a ZIP of faces for download ----
    zip_path = os.path.join(UPLOAD_FOLDER, f"{video_name}_faces.zip")
    face_count = 0
    if os.path.exists(faces_output_dir):
        face_files = [f for f in os.listdir(faces_output_dir) if f.endswith('.png')]
        face_count = len(face_files)
        if face_count > 0:
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for face_file in face_files:
                    face_full = os.path.join(faces_output_dir, face_file)
                    zf.write(face_full, arcname=face_file)
            message = f"Frames and faces extracted. Found {face_count} faces."
        else:
            message = "Frames extracted, but no faces were detected."
    else:
        message = "Frames extracted, but face cropping did not produce any output."

    return jsonify({'success': True, 'message': message, 'faces_zip': zip_path if os.path.exists(zip_path) else None})

# ---------- Page Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("/login")

@app.route("/auth")
def auth():
    return render_template("/auth")

@app.route("/howitworks")
def how():
    return render_template("/howitworks")

@app.route("/profile")
def profile():
    return render_template("/profile")

@app.route("/report")
def report():
    return render_template("/report")

@app.route("/upload")
def upload():
    return render_template("/upload")

@app.route("/index")
def index():
    return render_template("/index")

if __name__ == "__main__":
    app.run(debug=True, port=5000)