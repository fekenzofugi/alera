# Portaai FR

Based on `face_recogntion` library:
* <a href="https://github.com/ageitgey/face_recognition">Github Repository</a>

* <a href="https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78">Article</a>

# **Steps**:
1) Face Detection 
    * Using Histogram of Oriented Gradients (HOG) <a href="https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf" >Article</a>
2) Posing and Projecting Faces
    * Using Face Landmark Estimation <a href="http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf" >Article</a>
    * Affine Transformations
3) Encoding Faces
    * CNN-BASED face embedding.
4) Finding the person’s name from the encoding
    * Use a ML classifier to find the person in our database of known people who has the closest measurements to our target image.

## 1. Face Detection
### Histogram of Oriented Gradients(HOG):
HOG feature descriptor is a machine learning algorithm propose in 2005 and heavely used until these days.

## 2. Posing and Projecting Faces
### Face Landmark Estimation
The basic idea is we will come up with 68 specific points (called landmarks) that exist on every face — the top of the chin, the outside edge of each eye, the inner edge of each eyebrow, etc. Then we will train a machine learning algorithm to be able to find these 68 specific points on any face.
### Affine Tranformations
Simply rotate, scale and shear the image so that the eyes and mouth are centered as best as possible. We won’t do any fancy 3d warps because that would introduce distortions into the image. We are only going to use basic image transformations like rotation and scale that preserve parallel lines (called affine transformations).

## 3. Encoding Faces

### **Face Encoding with Deep Learning**  

1. **Need for Facial Measurements**  
   - Extracting key facial features is essential for recognition.  
   - Deep learning outperforms humans in determining which features matter.  
   - The goal is to generate **128 unique measurements** for each face.  
   - Machine learning people call the 128 measurements of each face an embedding <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf">FaceNet: A Unified Embedding for Face Recognition and Clustering</a>

2. **Triplet Training Approach**  
   - Load an image of a **known person**.  
   - Load another image of the **same person**.  
   - Load an image of a **different person**.  

3. **Network Adjustment**  
   - The model **brings closer** embeddings of the same person.  
   - It **pushes apart** embeddings of different people.  

4. **Repetition and Learning**  
   - **Millions of iterations** refine the model.  
   - The network learns to generate **consistent 128-dimensional embeddings**.  

5. **Computational Requirements**  
   - Training demands **high data and processing power**.  
   - Even with a **high-end GPU**, training takes **around 24 hours**.  

6. **Pre-Trained Models for Efficiency**  
   - Once trained, the network can **encode any new face**.  
   - **OpenFace** offers pre-trained models for direct use.  

7. **Interpreting Face Embeddings**  
   - The **128 values** represent facial features, though their exact meaning is unknown.  
   - The key is **consistent embeddings** for the same person across different images.

## 4. Finding the person’s name from the encoding
All we need to do is train a classifier that can take in the measurements from a new target image and tells which known person is the closest match. Running this classifier takes milliseconds. The result of the classifier is the name of the person!