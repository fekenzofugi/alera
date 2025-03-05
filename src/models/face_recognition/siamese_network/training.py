import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import Model
from keras.api.metrics import Recall, Precision


# Build Training Step Function
@tf.function
def train_step(batch, siamese_model, optimizer, binary_cross_loss):
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
def train(data, EPOCHS, siamese_model, optimizer, binary_cross_loss, checkpoint, checkpoint_prefix):
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
            loss = train_step(batch, siamese_model, optimizer, binary_cross_loss)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)