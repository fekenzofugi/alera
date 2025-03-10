import os
import os
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from flask_session import Session
import redis
import torch
from facenet_pytorch import InceptionResnetV1

from models.face_recognition.portaai_fr.classifier import FaceRecognitionClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

facenet_model = "vggface2"
workers = 0 if os.name == 'nt' else 4

classifier = FaceRecognitionClassifier()
resnet = InceptionResnetV1(pretrained=facenet_model).eval().to(device)

db = SQLAlchemy()
sess = Session()

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI="postgresql://portaai_user:portaai_password@db:5432/portaai_db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ECHO=False,
        SESSION_TYPE="redis",
        SESSION_PERMANENT=True,
        SESSION_USE_SIGNER=True,
        SESSION_REDIS=redis.from_url('redis://redis:6379'),
        PERMANENT_SESSION_LIFETIME=86400, 
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass  
    
    db.init_app(app)
    from flask_migrate import Migrate
    Migrate(app, db)
    app.config['SESSION_SQLALCHEMY'] = db
    sess.init_app(app)

    from .landing import landing_bp as landing_blueprint
    app.register_blueprint(landing_blueprint)

    from .auth import auth_bp as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from .main import main_bp as main_blueprint
    app.register_blueprint(main_blueprint)

    return app