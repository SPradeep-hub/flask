import os
import subprocess
import sys
import uuid
import zipfile
import re
import secrets
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

# ---------- Flask app initialization ----------
app = Flask(__name__,
            template_folder="app/templates",
            static_folder="app/static")

app.secret_key = secrets.token_hex(16)

# ---------- Path configuration ----------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
FRAMES_BASE   = os.path.join(BASE_DIR, 'frames')
FACES_BASE    = os.path.join(BASE_DIR, 'faces')
MODEL_PATH    = os.path.join(BASE_DIR, 'tmp_checkpoint', 'best_model.keras')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_BASE,   exist_ok=True)
os.makedirs(FACES_BASE,    exist_ok=True)

# ---------- Scripts ----------

FRAME_EXTRACTOR          = os.path.join(BASE_DIR, 'processing', '00-convert_video_to_image.py')
FACE_CROPPER             = os.path.join(BASE_DIR, 'processing', '01b-crop_faces_from_frames.py')
PREDICTOR                = os.path.join(BASE_DIR, 'processing', '05-predict_faces.py')
IMAGE_PREDICTOR          = os.path.join(BASE_DIR, 'processing', '06-predict_image.py')
AUDIO_PREDICTOR          = os.path.join(BASE_DIR, 'processing', '07-predict_audio.py')

# ---------- Allowed extensions ----------
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'webm', 'avi', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# ---------- Helper ----------
def run_script(script, args, timeout=180):
    """Run a python script, return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, script] + args,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)

def save_session(authenticity, face_count, verdict, filename, media_type):
    """Save result to Flask session."""
    session['authenticity'] = authenticity
    session['face_count']   = face_count
    session['verdict']      = verdict
    session['filename']     = filename
    session['media_type']   = media_type  # 'video', 'image', 'audio'

# ---------- VIDEO upload & predict endpoint ----------
@app.route("/upload_video", methods=['POST'])
def upload_video():
    print("=== VIDEO UPLOAD ENDPOINT HIT ===")

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({'success': False, 'error': f'Invalid file type: {ext}'}), 400

    unique_id     = uuid.uuid4().hex
    safe_filename = f"{unique_id}.{ext}"
    video_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(video_path)
    print(f"Video saved: {video_path}")

    frames_dir = os.path.join(FRAMES_BASE, unique_id)
    faces_dir  = os.path.join(FACES_BASE,  unique_id)

    # Step 1: Extract frames
    print("Extracting frames...")
    ok, out, err = run_script(FRAME_EXTRACTOR, [video_path, FRAMES_BASE], timeout=180)
    if not ok:
        return jsonify({'success': False, 'error': f'Frame extraction failed: {err}'}), 500

    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        return jsonify({'success': False, 'error': 'No frames were extracted from the video'}), 500

    print(f"Frames saved to: {frames_dir}")

    # Step 2: Crop faces
    print("Detecting and cropping faces...")
    ok, out, err = run_script(FACE_CROPPER, [frames_dir, faces_dir], timeout=180)
    if not ok:
        print(f"Face cropping warning: {err}")

    face_files = []
    if os.path.exists(faces_dir):
        face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith('.png')]
    face_count = len(face_files)
    print(f"{face_count} face(s) detected")

    if face_count == 0:
        save_session(None, 0, None, safe_filename, 'video')
        return jsonify({
            'success':      True,
            'message':      'Frames extracted but no faces were detected in the video.',
            'face_count':   0,
            'authenticity': None,
            'verdict':      None,
            'redirect':     '/report'
        })

    # Step 3: ZIP faces
    zip_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_faces.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in face_files:
            zf.write(os.path.join(faces_dir, f), arcname=f)

    # Step 4: Predict
    print("Running deepfake prediction...")
    ok, out, err = run_script(PREDICTOR, [faces_dir, MODEL_PATH], timeout=180)
    print("Predictor stdout:", out)
    if err:
        print("Predictor stderr:", err[:300])

    authenticity = None
    verdict      = None

    match = re.search(r'Authenticity\s*:\s*([\d.]+)%', out)
    if match:
        authenticity = round(float(match.group(1)), 2)
        verdict      = 'Likely REAL' if authenticity >= 50 else 'Likely FAKE'
        print(f"Authenticity: {authenticity}% -- {verdict}")
    else:
        print("Could not parse authenticity. Full stdout:", out)

    save_session(authenticity, face_count, verdict, safe_filename, 'video')

    return jsonify({
        'success':      True,
        'message':      f'Processed {face_count} face(s) from uploaded video.',
        'face_count':   face_count,
        'authenticity': authenticity,
        'verdict':      verdict,
        'faces_zip':    zip_path if os.path.exists(zip_path) else None,
        'redirect':     '/report'
    })


# ---------- IMAGE upload & predict endpoint ----------
@app.route("/upload_image", methods=['POST'])
def upload_image():
    print("=== IMAGE UPLOAD ENDPOINT HIT ===")

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({'success': False, 'error': f'Invalid file type: {ext}'}), 400

    unique_id     = uuid.uuid4().hex
    safe_filename = f"{unique_id}.{ext}"
    image_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(image_path)
    print(f"Image saved: {image_path}")

    print("Running image deepfake prediction...")
    ok, out, err = run_script(IMAGE_PREDICTOR, [image_path, MODEL_PATH], timeout=120)
    print("Predictor stdout:", out)
    if err:
        print("Predictor stderr:", err[:300])

    authenticity = None
    verdict      = None
    face_count   = 0

    auth_match    = re.search(r'Authenticity\s*:\s*([\d.]+)%', out)
    verdict_match = re.search(r'Verdict\s*:\s*(Likely REAL|Likely FAKE)', out)
    face_match    = re.search(r'Faces processed\s*:\s*(\d+)', out)
    no_face       = 'No face detected' in out

    if auth_match:
        authenticity = round(float(auth_match.group(1)), 2)
    if verdict_match:
        verdict = verdict_match.group(1)
    if face_match:
        face_count = int(face_match.group(1))
    if no_face:
        face_count = 0  # override — no real face found

    save_session(authenticity, face_count, verdict, safe_filename, 'image')

    return jsonify({
        'success':      True,
        'authenticity': authenticity,
        'verdict':      verdict,
        'face_count':   face_count,
        'redirect':     '/report'
    })


# ---------- AUDIO upload & predict endpoint ----------
@app.route("/upload_audio", methods=['POST'])
def upload_audio():
    print("=== AUDIO UPLOAD ENDPOINT HIT ===")

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({'success': False, 'error': f'Invalid file type: {ext}'}), 400

    unique_id     = uuid.uuid4().hex
    safe_filename = f"{unique_id}.{ext}"
    audio_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(audio_path)
    print(f"Audio saved: {audio_path}")

    print("Running audio deepfake prediction...")
    ok, out, err = run_script(AUDIO_PREDICTOR, [audio_path], timeout=300)
    print("Predictor stdout:", out)
    if err:
        print("Predictor stderr:", err[:300])

    authenticity  = None
    verdict       = None
    duration      = None

    auth_match     = re.search(r'Authenticity\s*:\s*([\d.]+)%', out)
    verdict_match  = re.search(r'Verdict\s*:\s*(Likely REAL|Likely FAKE)', out)
    duration_match = re.search(r'Duration\s*:\s*([\d.]+)s', out)

    if auth_match:
        authenticity = round(float(auth_match.group(1)), 2)
    if verdict_match:
        verdict = verdict_match.group(1)
    if duration_match:
        duration = float(duration_match.group(1))

    # face_count = 0 for audio (no faces, it's an audio clip)
    save_session(authenticity, 0, verdict, safe_filename, 'audio')

    return jsonify({
        'success':      True,
        'authenticity': authenticity,
        'verdict':      verdict,
        'duration':     duration,
        'redirect':     '/report'
    })


# ---------- Page Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/auth")
def auth():
    return render_template("auth.html")

@app.route("/howitworks")
def how():
    return render_template("howitworks.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/report")
def report():
    return render_template("report.html",
        authenticity = session.get('authenticity'),
        face_count   = session.get('face_count'),
        verdict      = session.get('verdict'),
        filename     = session.get('filename'),
        media_type   = session.get('media_type', 'video')
    )

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/index")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)