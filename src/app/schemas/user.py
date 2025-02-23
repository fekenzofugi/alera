from werkzeug.security import check_password_hash, generate_password_hash

from .. import db

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(64), unique=True, index=True)
    password = db.Column(db.String(256))

    def __init__(self, username, email, password):
        self.set_username(username)
        self.set_email(email)
        self.set_password(password)

    def set_username(self, username):
        self.username = username

    def get_username(self):
        return self.username

    def set_email(self, email):
        self.email = email

    def get_email(self):
        return self.email

    def set_password(self, password):
            self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f'<User {self.username}>'