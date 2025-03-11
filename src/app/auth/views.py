import functools
from ..schemas import User
from flask import (
    Blueprint, flash, redirect, render_template, request, url_for, session, Response
)
from app import db, facenet_model, workers, classifier, cap
import cv2
import torch
import os
import uuid
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import cv2

from models.face_recognition.portaai_fr.get_data import get_files_info
from models.face_recognition.portaai_fr.face_detection import detect_faces_mtcnn
from models.face_recognition.portaai_fr.face_embedding import collate_fn, get_image_embeddings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

global capture
capture=False

data_path = "/portaai/src/data"


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
                    os.makedirs(os.path.join(data_path, username), exist_ok=True)
                    session['regs_username'] = username
                except db.IntegrityError:
                    flash(f"User {username} is already registered.")
                    error = f"User {username} is already registered."

        if error:
            flash(error)
        else:
            return redirect(url_for('auth.face_register'))

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
    session.pop('regs_username', None)
    session.pop('num_files', None)
    return redirect(url_for('landing.index'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)
    return wrapped_view


@auth_bp.route('/face_register', methods=('GET', 'POST'))
def face_register():

    if session.get('regs_username') is None:
        return redirect(url_for('auth.register'))

    num_files = session.get('num_files', 0)
    # args_files = num_files
    # args_files = request.args.get('num_files', args_files, type=int)   
    # if args_files is not None:
    #     num_files = args_files

    if request.method=='POST':
        if request.form.get('click') == 'Continue':
            # Get Known Faces
            dataset = datasets.ImageFolder(data_path)
            dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
            aligned, names = detect_faces_mtcnn(dataset, loader)

            # Get Face Embeddings
            dataset_embeddings = get_image_embeddings(data_path, facenet_model, aligned)
            classifier.train(dataset_embeddings, names)  

            return redirect(url_for('auth.login'))          

    return render_template('auth/face_register.html', num_files=num_files)

@auth_bp.route('/auth_video_feed')
def video_feed():
    def generate(username):
        global capture

        while True:
            ret, frame = cap.read()
            if not ret:
                break           
            
            if capture:
                path = os.path.join(data_path, username)
                imgname = os.path.join(path, '{}.jpg'.format(uuid.uuid1()))
                # Write out anchor image
                cv2.imwrite(imgname, frame)
                capture = False

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    username = session.get('regs_username')
    return Response(generate(username), mimetype='multipart/x-mixed-replace; boundary=frame')

@auth_bp.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = True
            username = session.get('regs_username')
            # Wait a short time for image capture (ensures file gets written)
            import time
            time.sleep(1)
            num_files, _ = get_files_info(os.path.join(data_path, username))
            session['num_files'] = num_files  # Store updated count
            return redirect(url_for('auth.face_register'))
            
    return redirect(url_for('auth.face_register'))