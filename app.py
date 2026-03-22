import os
from flask import Flask, render_template, request

from processing.vidToImage import process_video

app = Flask(__name__, 
            template_folder="app/templates",
            static_folder="app/static",
)

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

if __name__ == "__main__":
    app.run(debug=True)