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
FRAME_EXTRACTOR = os.path.join(BASE_DIR, 'processing', '00-convert_video_to_image.py')
FACE_CROPPER    = os.path.join(BASE_DIR, 'processing', '01b-crop_faces_from_frames.py')
PREDICTOR       = os.path.join(BASE_DIR, 'processing', '05-predict_faces.py')

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'avi', 'mkv'}

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

# ---------- Upload & predict endpoint ----------
@app.route("/upload_video", methods=['POST'])
def upload_video():
    print("=== UPLOAD ENDPOINT HIT ===")

    # ── Validate file ─────────────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'success': False, 'error': f'Invalid file type: {ext}'}), 400

    # ── Save uploaded video ───────────────────────────────────────────────────
    unique_id     = uuid.uuid4().hex
    safe_filename = f"{unique_id}.{ext}"
    video_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(video_path)
    print(f"Video saved: {video_path}")

    frames_dir = os.path.join(FRAMES_BASE, unique_id)
    faces_dir  = os.path.join(FACES_BASE,  unique_id)

    # ── Step 1: Extract frames ────────────────────────────────────────────────
    print("Extracting frames...")
    ok, out, err = run_script(FRAME_EXTRACTOR, [video_path, FRAMES_BASE], timeout=180)
    if not ok:
        print(f"Frame extraction failed: {err}")
        return jsonify({'success': False, 'error': f'Frame extraction failed: {err}'}), 500

    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        return jsonify({'success': False, 'error': 'No frames were extracted from the video'}), 500

    print(f"Frames saved to: {frames_dir}")

    # ── Step 2: Crop faces ────────────────────────────────────────────────────
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
        session['authenticity'] = None
        session['face_count']   = 0
        session['verdict']      = None
        session['filename']     = safe_filename
        return jsonify({
            'success': True,
            'message': 'Frames extracted but no faces were detected in the video.',
            'face_count': 0,
            'authenticity': None,
            'verdict': None,
            'redirect': '/report'
        })

    # ── Step 3: ZIP faces ─────────────────────────────────────────────────────
    zip_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_faces.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in face_files:
            zf.write(os.path.join(faces_dir, f), arcname=f)

    # ── Step 4: Deepfake prediction ───────────────────────────────────────────
    print("Running deepfake prediction...")
    ok, out, err = run_script(PREDICTOR, [faces_dir, MODEL_PATH], timeout=180)
    print("Predictor stdout:", out)
    if err:
        print("Predictor stderr:", err[:300])

    # Parse authenticity
    authenticity = None
    verdict      = None

    match = re.search(r'Authenticity\s*:\s*([\d.]+)%', out)
    if match:
        authenticity = round(float(match.group(1)), 2)
        verdict = 'Likely REAL' if authenticity >= 50 else 'Likely FAKE'
        print(f"Authenticity: {authenticity}% -- {verdict}")
    else:
        print("Could not parse authenticity from predictor output")
        print("Full stdout was:", out)

    # ── Save to session ───────────────────────────────────────────────────────
    session['authenticity'] = authenticity
    session['face_count']   = face_count
    session['verdict']      = verdict
    session['filename']     = safe_filename

    # ── Response ──────────────────────────────────────────────────────────────
    return jsonify({
        'success':      True,
        'message':      f'Processed {face_count} face(s) from uploaded video.',
        'face_count':   face_count,
        'authenticity': authenticity,
        'verdict':      verdict,
        'faces_zip':    zip_path if os.path.exists(zip_path) else None,
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
        filename     = session.get('filename')
    )

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/index")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)