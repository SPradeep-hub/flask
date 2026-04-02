import os
from flask import Blueprint, Flask, current_app, render_template, request

from app import create_app
#from app.processing.video_processor import process_video    # pyright: ignore[reportMissingImports]

app = Flask(__name__, 
            template_folder="app/templates",
            static_folder="app/static",
)

app = create_app()
main = Blueprint('main', __name__)


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

if __name__ == "__main__":
    app.run(debug=True)