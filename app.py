import os
from flask import Blueprint, Flask, current_app, render_template, request

from app import create_app
#from app.processing.video_processor import process_video    # pyright: ignore[reportMissingImports]

app = Flask(__name__, 
            template_folder="app/templates",
            static_folder="app/static",
)

app = create_app()

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Login page
@app.route("/login")
def login():
    return render_template("login.html")

# Auth page 
@app.route("/auth")
def auth():
    return render_template("auth.html")

# How it works page
@app.route("/howitworks")
def how():
    return render_template("howitworks.html")

#profile page
@app.route("/profile")
def profile():
    return render_template("profile.html")

#report page
@app.route("/report")
def report():
    return render_template("report.html")

#upload page
@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/index")
def index():
    return render_template("index.html")


main = Blueprint('main', __name__)

# Configuration for allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Existing routes...
@main.route('/')
def index():
    return render_template('upload.html')   # or whatever your home page is

def jsonify(*args, **kwargs):
    raise NotImplementedError

def secure_filename(filename):
    raise NotImplementedError

# New route for video upload
@main.route('/upload_video', methods=['POST'])
def upload_video():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # Save temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)

    # Create a unique output folder for frames
    from datetime import datetime
    output_folder = os.path.join(
        current_app.config['FRAMES_FOLDER'],
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + os.path.splitext(filename)[0]
    )

    try:
        # Call your video processing function
        result_folder = process_video(temp_path, output_folder)

        # Count extracted frames
        frame_count = len([f for f in os.listdir(result_folder) if f.endswith('.png')])

        # Optional: delete the original uploaded video to save space
        # os.remove(temp_path)

        return jsonify({
            'success': True,
            'message': f'Video processed. Extracted {frame_count} frames.',
            'frames_folder': result_folder
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)