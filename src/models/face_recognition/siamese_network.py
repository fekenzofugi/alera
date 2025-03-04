import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import kagglehub


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
