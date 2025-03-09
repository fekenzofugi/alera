import os
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt


from get_data import get_data, get_files_info
from posing_projecting_faces import find_landmarks, align_image
from face_detection import histogram_of_oriented_gradients, detect_faces, detect_faces_mtcnn
from face_embedding import collate_fn, get_image_embeddings, tensor_to_image
from sklearn.metrics.pairwise import cosine_similarity
from classifier import FaceRecognitionClassifier

workers = 0 if os.name == 'nt' else 4

data_path = "images"
fernando_path = "images/fernando"

landmarks_model = "shape_predictor_68_face_landmarks.dat"
facenet_model = "vggface2" # or "casia-webface"
os.makedirs(data_path, exist_ok=True)
num_files, file_names = get_files_info(fernando_path)

# # Detect faces in the images
# for file_name in file_names:
    
#     img_path = os.path.join(fernando_path, file_name)
#     histogram_of_oriented_gradients(img_path, True)
#     detect_faces(img_path)

#     # You can download the required pre-trained face detection model here:
#     # https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
#     find_landmarks(img_path, landmarks_model)

#     face_aligned = align_image(img_path, landmarks_model)

# Get Cropped Faces
dataset = datasets.ImageFolder(data_path)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned, names = detect_faces_mtcnn(dataset, loader)

# Get Face Embeddings
dataset_embeddings = get_image_embeddings(data_path, facenet_model, aligned)
print(dataset_embeddings)


# Target Image
new_users_path = "target"
username = "user"
target_path = os.path.join(new_users_path, username)
os.makedirs(target_path, exist_ok=True)
get_data(target_path)

target_idx = 0
target_dataset = datasets.ImageFolder(new_users_path)
target_dataset.idx_to_class = {i:c for c, i in target_dataset.class_to_idx.items()}
target_loader = DataLoader(target_dataset, collate_fn=collate_fn, num_workers=workers)
target_aligned, target_names = detect_faces_mtcnn(target_dataset, target_loader)
target_img = tensor_to_image(target_aligned[target_idx])
target_embeddings = get_image_embeddings(new_users_path, facenet_model, target_aligned)
target_embedding = target_embeddings[target_idx]
target_name = target_names[target_idx]

# Calculate cosine similarity with the rest of the images
similarities = cosine_similarity([target_embedding], dataset_embeddings)[0]

# Plot the target image
fig, axes = plt.subplots(1, len(aligned) + 1, figsize=(15, 5))
axes[0].imshow(target_img)
axes[0].set_title(f"Target: {target_name}")
axes[0].axis('off')

# Plot the rest of the images with similarities
for i, (img, name) in enumerate(zip(aligned, names)):
    similarity = similarities[i]
    ax = axes[i + 1]
    ax.imshow(tensor_to_image(img))
    ax.set_title(f"Name: {name}\nSimilarity: {similarity:.2f}")
    ax.axis('off')

plt.suptitle(f"Target: {target_name}")
plt.show()

# Classify the target image
classifier = FaceRecognitionClassifier()
classifier.train(dataset_embeddings, names)
prediction = classifier.predict(target_embedding)

print(f"Prediction: {prediction}")