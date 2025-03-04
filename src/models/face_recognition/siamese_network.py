import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import kagglehub
import uuid

# Avoid Out Of Memory errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# Create directories
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# Collect Data: Negatives(# http://vis-www.cs.umass.edu/lfw/)
# Download latest version
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
labeled_path = os.path.join(path, 'lfw-deepfunneled/lfw-deepfunneled')

# Move LFW Images to the following repository data/negative
for directory in os.listdir(labeled_path):
    for file in os.listdir(os.path.join(labeled_path, directory)):
        EX_PATH = os.path.join(labeled_path, directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    # Collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

    # Print the number of jpg files in each directory
    num_anchors = len(os.listdir(ANC_PATH))
    num_positives = len(os.listdir(POS_PATH))
    num_negatives = len(os.listdir(NEG_PATH))
    print(f'Number of anchor images: {num_anchors}')
    print(f'Number of positive images: {num_positives}')
    print(f'Number of negative images: {num_negatives}')
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()

# Get Image Directories
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(300)


# Preprocessing - Scale & Resize
def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

# Create labeled dataset
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Build train and test partitions
def preprocesses_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# Build dataloader pipeline
data = data.map(preprocesses_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training Partition
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing Partition
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)