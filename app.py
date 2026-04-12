import os
import subprocess
import sys
import uuid
import zipfile
import re
import secrets
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt

load_dotenv()

# ---------- Flask app initialization ----------
app = Flask(__name__,
            template_folder="app/templates",
            static_folder="app/static")

app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
bcrypt = Bcrypt(app)

# ---------- MongoDB ----------
from pymongo import MongoClient

MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

mongo_client = MongoClient(MONGO_URI)
db           = mongo_client['deepverify']
users_col    = db['users']
history_col  = db['scan_history']
users_col.create_index('email', unique=True)
print("MongoDB Atlas connected!")

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
IMAGE_PREDICTOR = os.path.join(BASE_DIR, 'processing', '06-predict_image.py')
AUDIO_PREDICTOR = os.path.join(BASE_DIR, 'processing', '07-predict_audio.py')

# ---------- Allowed extensions ----------
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'webm', 'avi', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# ---------- Helpers ----------
def run_script(script, args, timeout=180):
    try:
        result = subprocess.run(
            [sys.executable, script] + args,
            capture_output=True, text=True,
            encoding='utf-8', errors='replace',
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)

def save_session(authenticity, face_count, verdict, filename, media_type):
    session['authenticity'] = authenticity
    session['face_count']   = face_count
    session['verdict']      = verdict
    session['filename']     = filename
    session['media_type']   = media_type

def save_history(filename, media_type, authenticity, verdict):
    """Save scan to MongoDB — only if user is logged in."""
    try:
        if session.get('user_email'):
            history_col.insert_one({
                'user_email':  session.get('user_email'),
                'user_name':   session.get('user_name', ''),
                'filename':    filename,
                'media_type':  media_type,
                'authenticity': authenticity,
                'verdict':     verdict,
                'analyzed_at': datetime.utcnow()
            })
            users_col.update_one(
                {'email': session.get('user_email')},
                {'$inc': {'scans': 1}}
            )
    except Exception as e:
        print(f"History save error (non-critical): {e}")

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


# ========== AUTH ROUTES ==========

@app.route('/api/signup', methods=['POST'])
def signup():
    data  = request.get_json()
    name  = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    pwd   = data.get('password', '')

    if not name or not email or not pwd:
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    if len(pwd) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    if users_col.find_one({'email': email}):
        return jsonify({'success': False, 'error': 'Email already registered'}), 409

    hashed = bcrypt.generate_password_hash(pwd).decode('utf-8')
    users_col.insert_one({
        'name':       name,
        'email':      email,
        'password':   hashed,
        'created_at': datetime.utcnow(),
        'scans':      0
    })

    session['user_email'] = email
    session['user_name']  = name
    return jsonify({'success': True, 'redirect': '/upload'})


@app.route('/api/login', methods=['POST'])
def api_login():
    data  = request.get_json()
    email = data.get('email', '').strip().lower()
    pwd   = data.get('password', '')

    if not email or not pwd:
        return jsonify({'success': False, 'error': 'All fields are required'}), 400

    user = users_col.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password'], pwd):
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

    session['user_email'] = email
    session['user_name']  = user.get('name', '')
    return jsonify({'success': True, 'redirect': '/upload'})


@app.route('/api/logout')
def api_logout():
    session.clear()
    return redirect('/')


# ========== VIDEO UPLOAD ==========

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

    print("Extracting frames...")
    ok, out, err = run_script(FRAME_EXTRACTOR, [video_path, FRAMES_BASE], timeout=180)
    if not ok:
        return jsonify({'success': False, 'error': f'Frame extraction failed: {err}'}), 500

    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        return jsonify({'success': False, 'error': 'No frames extracted from video'}), 500

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
        save_history(safe_filename, 'video', None, None)
        return jsonify({
            'success': True, 'message': 'No faces detected in video.',
            'face_count': 0, 'authenticity': None,
            'verdict': None, 'redirect': '/report'
        })

    zip_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_faces.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in face_files:
            zf.write(os.path.join(faces_dir, f), arcname=f)

    print("Running deepfake prediction...")
    ok, out, err = run_script(PREDICTOR, [faces_dir, MODEL_PATH], timeout=180)
    print("Predictor stdout:", out)

    authenticity = None
    verdict      = None
    match = re.search(r'Authenticity\s*:\s*([\d.]+)%', out)
    if match:
        authenticity = round(float(match.group(1)), 2)
        verdict      = 'Likely REAL' if authenticity >= 50 else 'Likely FAKE'

    save_session(authenticity, face_count, verdict, safe_filename, 'video')
    save_history(safe_filename, 'video', authenticity, verdict)

    return jsonify({
        'success': True, 'message': f'Processed {face_count} face(s).',
        'face_count': face_count, 'authenticity': authenticity,
        'verdict': verdict,
        'faces_zip': zip_path if os.path.exists(zip_path) else None,
        'redirect': '/report'
    })


# ========== IMAGE UPLOAD ==========

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

    authenticity  = None
    verdict       = None
    face_count    = 0

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
        face_count = 0

    save_session(authenticity, face_count, verdict, safe_filename, 'image')
    save_history(safe_filename, 'image', authenticity, verdict)

    return jsonify({
        'success': True, 'authenticity': authenticity,
        'verdict': verdict, 'face_count': face_count,
        'redirect': '/report'
    })


# ========== AUDIO UPLOAD ==========

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

    save_session(authenticity, 0, verdict, safe_filename, 'audio')
    save_history(safe_filename, 'audio', authenticity, verdict)

    return jsonify({
        'success': True, 'authenticity': authenticity,
        'verdict': verdict, 'duration': duration,
        'redirect': '/report'
    })


# ========== PROFILE API ==========

@app.route('/api/profile')
def api_profile():
    if 'user_email' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401

    user = users_col.find_one(
        {'email': session['user_email']},
        {'password': 0, '_id': 0}
    )
    history = list(history_col.find(
        {'user_email': session['user_email']},
        {'_id': 0}
    ).sort('analyzed_at', -1).limit(20))

    for h in history:
        if 'analyzed_at' in h:
            h['analyzed_at'] = h['analyzed_at'].strftime('%Y-%m-%d %H:%M')

    return jsonify({
        'success': True,
        'user':    user,
        'history': history
    })


# ========== PAGE ROUTES ==========

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