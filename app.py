import os

from flask import Flask, render_template, request

from processing.vidToImage import process_video

app = Flask(__name__, 
            template_folder="app/templates",
            static_folder="app/static",
            UPLOAD_FOLDER = "uploads",
            OUTPUT_FOLDER = "output_folder")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Login page
@app.route("/login")
def login():
    return render_template("login.html")

# Auth page (if needed)
@app.route("/auth")
def auth():
    return render_template("auth.html")

# How it works page
@app.route("/how-it-works")
def how():
    return render_template("howitworks.html")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["video"]

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        output_path = os.path.join(OUTPUT_FOLDER, file.filename.split('.')[0])

        process_video(video_path, output_path)

        return "Frames extracted successfully!"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)