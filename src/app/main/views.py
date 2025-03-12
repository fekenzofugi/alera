from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session, Response)
from ..schemas import User, Chat
from app import db, facenet_model, workers, classifier, resnet, cap
from ..auth.views import login_required
from utils.tts_v1 import text_to_speech_v1
import requests
import sys
import os
import json
sys.path.append('../')

from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch
import cv2
from facenet_pytorch import MTCNN

from models.face_recognition.portaai_fr.get_data import get_files_info
from models.face_recognition.portaai_fr.face_detection import detect_faces_mtcnn
from models.face_recognition.portaai_fr.face_embedding import collate_fn, get_image_embeddings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

main_bp = Blueprint('main', __name__)

data_path = "/portaai/src/data"
num_dirs = len([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
print(f"Number of directories in {data_path}: {num_dirs}")

if num_dirs > 1:
    # Get Known Faces
    dataset = datasets.ImageFolder(data_path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    aligned, names = detect_faces_mtcnn(dataset, loader)

    # Get Face Embeddings
    dataset_embeddings = get_image_embeddings(data_path, facenet_model, aligned)
    classifier.train(dataset_embeddings, names)

from flask import Response, stream_with_context

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
                messages.append({"role": "user", "content": chat.user_input})
                messages.append({"role": "assistant", "content": chat.text})

        messages.append({"role": "user", "content": user_input})

        def generate_text():
            try:
                response = requests.post(
                    'http://ollama:11434/api/chat', 
                    json={"messages": messages, "stream": True, "model": "myllama3"},
                    stream=True
                )

                collected_text = ""

                for chunk in response.iter_lines():
                    if chunk:
                        chunk_data = chunk.decode('utf-8')
                        chunk_data = json.loads(chunk_data)["message"]["content"]
                        yield chunk_data
                        collected_text += chunk_data
                chat_title = f"audio{int(User.query.get(session['user_id']).chats.count())}"
                # Save chat to database
                chat = Chat(author_id=session['user_id'], title=chat_title, text=collected_text, user_input=user_input)
                db.session.add(chat)
                db.session.commit()

                # Convert text to speech
                num_chats = User.query.get(session['user_id']).chats.count()
                chat_title = f"audio{int(num_chats) - 1}"
                text_to_speech_v1(collected_text, audio_output_path, chat_title)

            except Exception as e:
                yield f"Error: {str(e)}"

        return Response(stream_with_context(generate_text()), content_type='text/plain')

    return render_template('main/index.html', res=res, user_input=user_input, audio_filename=f"{chat_title}.mp3")



@main_bp.route('/video_feed')
def video_feed():
    def generate():

        mtcnn = MTCNN(
            margin=0, min_face_size=50,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device, keep_all=True
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    # Convert frame to RGB and then to tensor
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mg_cropped = mtcnn(rgb_frame)
                    if mg_cropped is not None:
                        mg_cropped = mg_cropped.to(device)
                        img_embedding = resnet(mg_cropped).detach().cpu()
                        prediction, probabilities = classifier.predict(img_embedding[0])
                    
                        color = (0, 255, 0) if probabilities >= 0.6 else (0, 0, 255)
                        cv2.rectangle(frame, 
                                      (int(box[0]), int(box[1])), 
                                      (int(box[2]), int(box[3])), 
                                      color, 
                                      2)
                        cv2.putText(frame, 
                                    f'{prediction} ({probabilities:.2f})', 
                                    (int(box[0]), int(box[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    color, 
                                    2, 
                                    cv2.LINE_AA)
                    

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')