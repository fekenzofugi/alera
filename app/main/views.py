from flask import (
    Blueprint, render_template, request, redirect, url_for, flash)
import sys
from ..utils.chain import model
sys.path.append('../')

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=('GET', 'POST'))
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form['user_input']
        print(f"User input: {user_input}")
        try:
            response = model.invoke(input=user_input)
        except Exception as e:
            flash(f"Error connecting to the model service: {e}")
        print(f"response: {response}")
        return redirect(url_for("main.index"))
    return render_template('main/index.html', response=response)