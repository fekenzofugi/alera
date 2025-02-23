import functools
from ..schemas import User
from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, session
)
from app import db

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form.get('email')  # Use .get() para evitar KeyError

        error = None
        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            existing_user = db.session.query(User).filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                error = 'Username or email is already registered.'
            else:
                try:
                    user = User(username=username, email=email, password=password)
                    db.session.add(user)
                    db.session.commit()
                except db.IntegrityError:
                    flash(f"User {username} is already registered.")
                    error = f"User {username} is already registered."

        if error:
            flash(error)
        else:
            return redirect(url_for('auth.login'))

    return render_template('auth/register.html')

@auth_bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        user = db.session.query(User).filter_by(username=username).first()
        if user is None:
            error = 'Incorrect credentials.'
        elif not user.check_password(password=password):
            error = 'Incorrect credentials.'

        if error is None:
            session['user_id'] = user.id
            session['username'] = user.username

            return redirect(url_for('main.index'))
        elif "Cannot authenticate" in error:
            error = 'Try again'
        flash(error)
    return render_template('auth/login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('landing.index'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)
    return wrapped_view