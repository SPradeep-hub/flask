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

# Auth page (if needed)
@app.route("/auth")
def auth():
    return render_template("auth.html")

# How it works page
@app.route("/how-it-works")
def how():
    return render_template("howitworks.html")

if __name__ == "__main__":
    app.run(debug=True)