import os
import subprocess
import sys
import uuid
import zipfile
import re
import secrets
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask import send_from_directory

load_dotenv()

# ---------- Flask app initialization ----------
app = Flask(__name__,
            template_folder="app/templates",
            static_folder="app/static")

app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
bcrypt = Bcrypt(app)

# ---------- MongoDB ----------
import certifi
from pymongo import MongoClient

MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

mongo_client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=20000,
    socketTimeoutMS=20000
)
db          = mongo_client['deepverify']
users_col   = db['users']
history_col = db['scan_history']

try:
    mongo_client.admin.command('ping')
    users_col.create_index('email', unique=True)
    print("MongoDB Atlas connected!")
except Exception as e:
    print(f"MongoDB connection error: {e}")
    raise

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
IMAGE_PREDICTOR = os.path.join(BASE_DIR, 'processing', '06-predict_image.py')
AUDIO_PREDICTOR = os.path.join(BASE_DIR, 'processing', '07-predict_audio.py')

# ---------- Allowed extensions ----------
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'webm', 'avi', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# ---------- Load model ONCE at startup ----------
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

print("Loading deepfake detection model...")
try:
    DEEPFAKE_MODEL = load_model(MODEL_PATH)
    print("Model loaded and ready!")
except Exception as e:
    print(f"Model load error: {e}")
    DEEPFAKE_MODEL = None

# ---------- In-process batch prediction ----------
def predict_faces_inprocess(faces_dir: str) -> dict:
    """
    Run batch prediction using the already-loaded model.
    No subprocess overhead — much faster than calling 05-predict_faces.py.
    """
    if DEEPFAKE_MODEL is None:
        return {'success': False, 'error': 'Model not loaded'}

    if not os.path.isdir(faces_dir):
        return {'success': False, 'error': f'Faces directory not found: {faces_dir}'}

    SUPPORTED = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    try:
        image_files = [
            f for f in os.listdir(faces_dir)
            if f.lower().endswith(SUPPORTED)
        ]
    except Exception as e:
        return {'success': False, 'error': f'Unable to read faces directory: {e}'}

    if not image_files:
        return {'success': False, 'error': 'No images found in faces folder'}

    # Load all images into a single batch
    batch       = []
    valid_names = []

    for fname in image_files:
        try:
            img = load_img(os.path.join(faces_dir, fname), target_size=(224, 224))
            arr = img_to_array(img)
            batch.append(arr)
            valid_names.append(fname)
        except Exception as e:
            print(f"Skipped {fname}: {e}")

    if not batch:
        return {'success': False, 'error': 'All face images failed to load'}

    # Predict entire batch at once — faster than one-by-one
    try:
        batch_arr = preprocess_input(np.array(batch))
        raw_preds = DEEPFAKE_MODEL.predict(batch_arr, verbose=0).flatten()
    except Exception as e:
        return {'success': False, 'error': f'Model prediction failed: {e}'}

    per_face = [round((1 - float(p)) * 100, 2) for p in raw_preds]
    avg_fake = float(np.mean(raw_preds))
    auth     = round((1 - avg_fake) * 100, 2)
    verdict  = 'Likely REAL' if auth >= 50 else 'Likely FAKE'

    for fname, pred, score in zip(valid_names, raw_preds, per_face):
        label = 'FAKE' if pred > 0.5 else 'REAL'
        print(f"  {fname}: fake_prob={pred:.4f} ({label}) | auth={score:.1f}%")

    return {
        'success':      True,
        'authenticity': auth,
        'verdict':      verdict,
        'face_count':   len(raw_preds),
        'per_face':     per_face,
        'raw_probs':    [float(p) for p in raw_preds]
    }

# ---------- Helpers ----------
def run_script(script, args, timeout=180):
    if not os.path.isfile(script):
        return False, "", f"Script not found: {script}"

    try:
        result = subprocess.run(
            [sys.executable, script] + args,
            capture_output=True, text=True,
            encoding='utf-8', errors='replace',
            timeout=timeout
        )
        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip() or f"Script exited with code {result.returncode}"
            return False, result.stdout, error_output
        return True, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)

def json_error(message, status_code=500):
    return jsonify({'success': False, 'error': message}), status_code

def get_request_json():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be valid JSON")
    return data

def get_uploaded_file(field_name, allowed_extensions):
    if field_name not in request.files:
        raise ValueError('No file part')

    file = request.files[field_name]
    if not file or file.filename == '':
        raise ValueError('No selected file')

    cleaned_name = secure_filename(file.filename)
    if not cleaned_name:
        raise ValueError('Invalid filename')

    ext = cleaned_name.rsplit('.', 1)[-1].lower() if '.' in cleaned_name else ''
    if ext not in allowed_extensions:
        raise ValueError(f'Invalid file type: {ext}')

    return file, ext

def save_session(authenticity, face_count, verdict, filename, media_type, per_face=None):
    session['authenticity'] = authenticity
    session['face_count']   = face_count
    session['verdict']      = verdict
    session['filename']     = filename
    session['media_type']   = media_type
    session['per_face']     = per_face or []

def save_history(filename, media_type, authenticity, verdict):
    """Save scan to MongoDB — only if user is logged in."""
    try:
        if session.get('user_email'):
            history_col.insert_one({
                'user_email':   session.get('user_email'),
                'user_name':    session.get('user_name', ''),
                'filename':     filename,
                'media_type':   media_type,
                'authenticity': authenticity,
                'verdict':      verdict,
                'analyzed_at':  datetime.utcnow()
            })
            users_col.update_one(
                {'email': session.get('user_email')},
                {'$inc': {'scans': 1}}
            )
    except Exception as e:
        print(f"History save error (non-critical): {e}")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.exception("Unhandled application error")
    wants_json = request.path.startswith('/api/') or request.path.startswith('/upload_')
    if wants_json:
        return json_error("An unexpected server error occurred.")
    return "An unexpected server error occurred.", 500

# ========== AUTH ROUTES ==========

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data  = get_request_json()
        name  = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        pwd   = data.get('password', '')

        if not name or not email or not pwd:
            return json_error('All fields are required', 400)
        if len(pwd) < 8:
            return json_error('Password must be at least 8 characters', 400)
        if users_col.find_one({'email': email}):
            return json_error('Email already registered', 409)

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
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        app.logger.exception("Signup failed")
        return json_error(f"Signup failed: {e}")


@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data  = get_request_json()
        email = data.get('email', '').strip().lower()
        pwd   = data.get('password', '')

        if not email or not pwd:
            return json_error('All fields are required', 400)

        user = users_col.find_one({'email': email})
        if not user or not bcrypt.check_password_hash(user['password'], pwd):
            return json_error('Invalid email or password', 401)

        session['user_email'] = email
        session['user_name']  = user.get('name', '')
        return jsonify({'success': True, 'redirect': '/upload'})
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        app.logger.exception("Login failed")
        return json_error(f"Login failed: {e}")


@app.route('/api/logout')
def api_logout():
    try:
        session.clear()
    except Exception:
        app.logger.exception("Logout failed")
    return redirect('/')


# ========== VIDEO UPLOAD ==========

@app.route("/upload_video", methods=['POST'])
def upload_video():
    print("=== VIDEO UPLOAD ENDPOINT HIT ===")

    try:
        file, ext = get_uploaded_file('file', ALLOWED_VIDEO_EXTENSIONS)

        unique_id     = uuid.uuid4().hex
        safe_filename = f"{unique_id}.{ext}"
        video_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(video_path)
        print(f"Video saved: {video_path}")

        frames_dir = os.path.join(FRAMES_BASE, unique_id)
        faces_dir  = os.path.join(FACES_BASE, unique_id)

        print("Extracting frames...")
        ok, out, err = run_script(FRAME_EXTRACTOR, [video_path, FRAMES_BASE], timeout=180)
        if not ok:
            return json_error(f'Frame extraction failed: {err}')

        if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
            return json_error('No frames extracted from video')

        print("Detecting and cropping faces...")
        ok, out, err = run_script(FACE_CROPPER, [frames_dir, faces_dir], timeout=180)
        if not ok:
            app.logger.warning("Face cropping warning: %s", err)

        face_files = []
        if os.path.exists(faces_dir):
            face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith('.png')]
        face_count = len(face_files)
        print(f"{face_count} face(s) detected")

        if face_count == 0:
            save_session(None, 0, None, safe_filename, 'video', [])
            save_history(safe_filename, 'video', None, None)
            return jsonify({
                'success': True, 'message': 'No faces detected in video.',
                'face_count': 0, 'authenticity': None,
                'verdict': None, 'redirect': '/report'
            })

        zip_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_faces.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for face_file in face_files:
                zf.write(os.path.join(faces_dir, face_file), arcname=face_file)

        print("Running deepfake prediction (in-process batch)...")
        result = predict_faces_inprocess(faces_dir)
        if not result['success']:
            return json_error(f"Prediction failed: {result['error']}")

        authenticity = result['authenticity']
        verdict      = result['verdict']
        face_count   = result['face_count']
        per_face     = result['per_face']

        save_session(authenticity, face_count, verdict, safe_filename, 'video', per_face)
        save_history(safe_filename, 'video', authenticity, verdict)

        return jsonify({
            'success':      True,
            'authenticity': authenticity,
            'verdict':      verdict,
            'face_count':   face_count,
            'per_face':     per_face,
            'redirect':     '/report'
        })
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        app.logger.exception("Video upload failed")
        return json_error(f"Video upload failed: {e}")


# ========== IMAGE UPLOAD ==========

@app.route("/upload_image", methods=['POST'])
def upload_image():
    print("=== IMAGE UPLOAD ENDPOINT HIT ===")

    try:
        file, ext = get_uploaded_file('file', ALLOWED_IMAGE_EXTENSIONS)

        unique_id     = uuid.uuid4().hex
        safe_filename = f"{unique_id}.{ext}"
        image_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(image_path)
        print(f"Image saved: {image_path}")

        print("Running image deepfake prediction...")
        ok, out, err = run_script(IMAGE_PREDICTOR, [image_path, MODEL_PATH], timeout=120)
        if not ok:
            return json_error(f'Image prediction failed: {err}')

        print("Predictor stdout:", out)

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
            face_count = 0

        save_session(authenticity, face_count, verdict, safe_filename, 'image', [])
        save_history(safe_filename, 'image', authenticity, verdict)

        return jsonify({
            'success':      True,
            'authenticity': authenticity,
            'verdict':      verdict,
            'face_count':   face_count,
            'redirect':     '/report'
        })
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        app.logger.exception("Image upload failed")
        return json_error(f"Image upload failed: {e}")


# ========== AUDIO UPLOAD ==========

@app.route("/upload_audio", methods=['POST'])
def upload_audio():
    print("=== AUDIO UPLOAD ENDPOINT HIT ===")

    try:
        file, ext = get_uploaded_file('file', ALLOWED_AUDIO_EXTENSIONS)

        unique_id     = uuid.uuid4().hex
        safe_filename = f"{unique_id}.{ext}"
        audio_path    = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(audio_path)
        print(f"Audio saved: {audio_path}")

        print("Running audio deepfake prediction...")
        ok, out, err = run_script(AUDIO_PREDICTOR, [audio_path], timeout=300)
        if not ok:
            return json_error(f'Audio prediction failed: {err}')

        print("Predictor stdout:", out)

        authenticity = None
        verdict      = None
        duration     = None

        auth_match     = re.search(r'Authenticity\s*:\s*([\d.]+)%', out)
        verdict_match  = re.search(r'Verdict\s*:\s*(Likely REAL|Likely FAKE)', out)
        duration_match = re.search(r'Duration\s*:\s*([\d.]+)s', out)

        if auth_match:
            authenticity = round(float(auth_match.group(1)), 2)
        if verdict_match:
            verdict = verdict_match.group(1)
        if duration_match:
            duration = float(duration_match.group(1))

        save_session(authenticity, 0, verdict, safe_filename, 'audio', [])
        save_history(safe_filename, 'audio', authenticity, verdict)

        return jsonify({
            'success':      True,
            'authenticity': authenticity,
            'verdict':      verdict,
            'duration':     duration,
            'redirect':     '/report'
        })
    except ValueError as e:
        return json_error(str(e), 400)
    except Exception as e:
        app.logger.exception("Audio upload failed")
        return json_error(f"Audio upload failed: {e}")


# ========== PROFILE API ==========

@app.route('/api/profile')
def api_profile():
    try:
        if 'user_email' not in session:
            return json_error('Not logged in', 401)

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
    except Exception as e:
        app.logger.exception("Profile fetch failed")
        return json_error(f"Profile fetch failed: {e}")


# ========== FAVICON ==========

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'app', 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


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

@app.route("/gta")
def gta12():
    return render_template("gta12.html")

@app.route("/report")
def report():
    return render_template("report.html",
        authenticity = session.get('authenticity'),
        face_count   = session.get('face_count'),
        verdict      = session.get('verdict'),
        filename     = session.get('filename'),
        media_type   = session.get('media_type', 'video'),
        per_face     = session.get('per_face', [])
    )

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/index")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
