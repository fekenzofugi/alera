import tensorflow as tf


# Preprocessing - Scale & Resize
def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 105x105x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


# Build train and test partitions
def preprocesses_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)