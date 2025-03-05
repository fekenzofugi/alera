import tensorflow as tf
from keras.api.models import Model
from keras.api.metrics import Recall, Precision
from keras.api.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


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
def make_siamese_model(embedding): 
    
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