from flask import Flask, request, jsonify, render_template
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

from get_data import get_data, get_files_info
from posing_projecting_faces import find_landmarks, align_image
from face_detection import histogram_of_oriented_gradients, detect_faces, detect_faces_mtcnn
from face_embedding import collate_fn, get_image_embeddings, tensor_to_image
from sklearn.metrics.pairwise import cosine_similarity
from classifier import FaceRecognitionClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

app = Flask(__name__)
data_path = "images"
landmarks_model = "shape_predictor_68_face_landmarks.dat"
facenet_model = "vggface2"
target_path = "target"
workers = 0 if os.name == 'nt' else 4
classifier = FaceRecognitionClassifier()
resnet = InceptionResnetV1(pretrained=facenet_model).eval().to(device)

# Get Known Faces
dataset = datasets.ImageFolder(data_path)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned, names = detect_faces_mtcnn(dataset, loader)

# Get Face Embeddings
dataset_embeddings = get_image_embeddings(data_path, facenet_model, aligned)
classifier.train(dataset_embeddings, names)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        
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
        cap.release()

    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    