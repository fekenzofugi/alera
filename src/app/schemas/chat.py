from werkzeug.security import check_password_hash, generate_password_hash

from .. import db

class Chat(db.Model):
    __tablename__ = 'chat'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(128), nullable=False)
    text = db.Column(db.Text, nullable=False)
    user_input = db.Column(db.Text, nullable=False)

    def __init__(self, author_id, title, text, user_input):
        self.author_id = author_id
        self.title = title
        self.text = text
        self.user_input = user_input

    def __repr__(self):
        return f'<Chat {self.title}>'
