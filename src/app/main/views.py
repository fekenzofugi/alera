from flask import (
    Blueprint, render_template, request, redirect, url_for, flash)
from ..auth.views import login_required
import requests
import sys
sys.path.append('../')

main_bp = Blueprint('main', __name__)

@main_bp.route('/llm', methods=('GET', 'POST'))
@login_required
def index():
    res=None
    if request.method == "POST":
        user_input = request.form['user_input']
        print(f"User input: {user_input}")
        try:
            res = requests.post('http://ollama:11434/api/generate', json={
                "prompt": user_input,
                "stream" : False,
                "model" : "myllama3"
            }).json()
        except Exception as e:
            flash(f"Error connecting to the model service: {e}")
        return render_template('main/index.html', res=res)
    return render_template('main/index.html', res=res)