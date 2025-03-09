from sklearn import svm
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FaceRecognitionClassifier:
    def __init__(self):
        self.model = svm.SVC(kernel='linear', probability=True)

    def train(self, embeddings, labels):
        start_time = time.time()
        
        self.model.fit(embeddings, labels)
        
        end_time = time.time()
        print(f"Training took {end_time - start_time} seconds")

    def predict(self, target_embedding):
        start_time = time.time()
        
        prediction = self.model.predict([target_embedding])
        probabilities = self.model.predict_proba([target_embedding])
        
        end_time = time.time()
        # print(f"Prediction took {end_time - start_time} seconds")
        
        if max(probabilities[0]) < 0.6:
            return "unknown", max(probabilities[0])
        
        return prediction[0], max(probabilities[0])