import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.metrics import Recall, Precision

from get_data import get_data
from preprocessing import preprocesses_twin
from model import make_embedding, make_siamese_model, L1Dist
from training import train


# Avoid Out Of Memory errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# Load Data
anchor, positive, negative = get_data()

# Create labeled dataset
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

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
embedding = make_embedding()
print(embedding.summary())

siamese_model = make_siamese_model(embedding)
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

# Train the model
EPOCH = 12
train(train_data, EPOCH, siamese_model, optimizer, binary_cross_loss, checkpoint, checkpoint_prefix)

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