from flask import Blueprint, request, jsonify, session, redirect, url_for
from flask_bcrypt import Bcrypt
from database import users_col, history_col
from datetime import datetime
import re

auth = Blueprint('auth', __name__)
bcrypt = Bcrypt()

def is_valid_email(email):
    return re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email)

@auth.route('/api/signup', methods=['POST'])
def signup():
    data  = request.get_json()
    name  = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    pwd   = data.get('password', '')

    if not name or not email or not pwd:
        return jsonify({'success': False, 'error': 'All fields required'}), 400
    if not is_valid_email(email):
        return jsonify({'success': False, 'error': 'Invalid email'}), 400
    if len(pwd) < 8:
        return jsonify({'success': False, 'error': 'Password must be 8+ characters'}), 400
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


@auth.route('/api/login', methods=['POST'])
def login():
    data  = request.get_json()
    email = data.get('email', '').strip().lower()
    pwd   = data.get('password', '')

    if not email or not pwd:
        return jsonify({'success': False, 'error': 'All fields required'}), 400

    user = users_col.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password'], pwd):
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

    session['user_email'] = email
    session['user_name']  = user['name']
    return jsonify({'success': True, 'redirect': '/upload'})


@auth.route('/api/logout')
def logout():
    session.clear()
    return redirect('/')


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated
