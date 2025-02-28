from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session)
from ..schemas import User, Chat
from .. import db
from ..auth.views import login_required
from utils.tts_v1 import text_to_speech_v1
import requests
import sys
import os
sys.path.append('../')

main_bp = Blueprint('main', __name__)

@main_bp.route('/llm', methods=('GET', 'POST'))
@login_required
def index():
    audio_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    newest_chat = Chat.query.filter_by(author_id=session['user_id']).order_by(Chat.id.desc()).first()
    
    if newest_chat:
        chat_title = newest_chat.title
        user_input = newest_chat.user_input
        res = newest_chat.text
    else:
        chat_title = 'audio0'
        user_input = None
        res = None

    if request.method == "POST":
        user_input = request.form['user_input']
        print(f"User input: {user_input}")
        messages = []
        if newest_chat:
            for chat in User.query.get(session['user_id']).chats.all():
                messages.append({
                    "role": "user",
                    "content": chat.user_input
                })
                messages.append({
                    "role": "assistant",
                    "content": chat.text
                })
        try:
            messages.append({
                "role": "user",
                "content": user_input
            })
            response = requests.post('http://ollama:11434/api/chat', json={
                "messages": messages,
                "stream" : False,
                "model" : "myllama3",
            }).json()
            chat = Chat(author_id=session['user_id'], title=chat_title, text=response['message']['content'], user_input=user_input)
            db.session.add(chat)
            db.session.commit()
            num_chats = User.query.get(session['user_id']).chats.count()
            chat_title = f"audio{int(num_chats) - 1}"
            text_to_speech_v1(response['message']['content'], audio_output_path, chat_title)
            res = response['message']['content']
            flash(f"{chat_title} saved successfully. Audio output path: {audio_output_path}. Number of files in audio output path: {len(os.listdir(audio_output_path))}. Files: {os.listdir(audio_output_path)}, {res}")
        except Exception as e:
            flash(f"Error connecting to the model service: {e}")
        return render_template('main/index.html', res=res, user_input=user_input, audio_filename=f"{chat_title}.mp3")
    return render_template('main/index.html', res=res, user_input=user_input, audio_filename=f"{chat_title}.mp3")
