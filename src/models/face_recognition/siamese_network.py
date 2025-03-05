import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import Model
from keras.api.metrics import Recall, Precision
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

# # Establish a connection to the webcam
# cap = cv2.VideoCapture(0)
# while cap.isOpened(): 
#     ret, frame = cap.read()
   
#     # Cut down frame to 250x250px
#     frame = frame[120:120+250,200:200+250, :]
    
#     # Collect anchors 
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         # Create the unique file path 
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # Write out anchor image
#         cv2.imwrite(imgname, frame)
    
#     # Collect positives
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#         # Create the unique file path 
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # Write out positive image
#         cv2.imwrite(imgname, frame)
    
#     # Show image back to screen
#     cv2.imshow('Image Collection', frame)
    
#     # Breaking gracefully
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#         break

#     # Print the number of jpg files in each directory
#     num_anchors = len(os.listdir(ANC_PATH))
#     num_positives = len(os.listdir(POS_PATH))
#     num_negatives = len(os.listdir(NEG_PATH))
#     print(f'Number of anchor images: {num_anchors}')
#     print(f'Number of positive images: {num_positives}')
#     print(f'Number of negative images: {num_negatives}')
        
# # Release the webcam
# cap.release()
# # Close the image show frame
# cv2.destroyAllWindows()

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
    
    # Preprocessing steps - resizing the image to be 105x105x3
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

# Model Engineering

# Build Embedding Layer
def make_embedding():
    input_layer = Input(shape=(105,105,3), name="input_image")

    # First Block
    conv_layer1 = Conv2D(filters=64, kernel_size=(10,10), strides=(1,1), activation='relu', padding="valid")(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(conv_layer1)

    # Second Block
    conv_layer2 = Conv2D(filters=128, kernel_size=(7,7), strides=(1,1), activation='relu', padding="valid")(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(conv_layer2)

    # Third Block
    conv_layer3 = Conv2D(filters=128, kernel_size=(4,4), strides=(1,1), activation='relu', padding="valid")(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(conv_layer3)

    # Final Embedding Block
    conv_layer4 = Conv2D(filters=256, kernel_size=(4,4), strides=(1,1), activation='relu', padding="valid")(max_pool3)
    flatten_layer = Flatten()(conv_layer4)
    dense_layer1 = Dense(units=4096, activation='sigmoid')(flatten_layer)


    return Model(inputs=[input_layer], outputs=[dense_layer1], name="embedding")

embedding = make_embedding()
print(embedding.summary())

# Build Distance Layer
# Siamese L1 Distance class
class L1Dist(Layer):
    
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)  # Ensure it's a tensor
        validation_embedding = tf.convert_to_tensor(validation_embedding)  # Ensure it's a tensor
        return tf.math.abs(input_embedding - validation_embedding)


# Make Siamese Network
def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105, 105,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(105,105,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()

print(siamese_model.summary())

# Training
# Setup Loss and Optimizer
binary_cross_loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Establish Checkpoints
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=siamese_model)

# Build Training Step Function
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]

        yhat = siamese_model(X, training=True)

        # Reshape yhat to match y
        yhat = tf.reshape(yhat, (-1,))

        loss = binary_cross_loss(y, yhat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss


# Build training loop
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)

# Train the model
EPOCH = 12
train(train_data, EPOCH)

# Evaluation
# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])[0]
print(y_hat)

# Post processing the results
results = [] 
for prediction in y_hat:
    if prediction > 0.5:
        print('Positive')
        results.append(1)
    else:
        print('Negative')
        results.append(0)
print(results)

# Compare with true labels
print(y_true)


# Calculate Recall and Precision
# Creating a metric object 
m = Recall()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
print(m.result().numpy())

# Creating a metric object 
m = Precision()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
print(m.result().numpy())


r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())

# Visualize the results

# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()

# Save the model
# Save Weights
siamese_model.save('siamese_model.h5')

# Reload the model
reloaded_siemese_model = tf.keras.models.load_model(
    'siamese_model.h5',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy}
)

# Check the model
print(reloaded_siemese_model.summary())

print(reloaded_siemese_model.predict([test_input, test_val]))