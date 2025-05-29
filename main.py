import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.utils import to_categorical



""" 
    Data Preprocessing
    
    1) Normalize the pixel values to be between 0 and 1.
    2) Reshape the images to be flat vectors of size 784 (28x28).
    3) Convert the labels to one-hot encoded format.
    4) Split the training data into training and validation sets.
    5) Use the Adam optimizer for training.
    6) Use categorical crossentropy as the loss function.
"""

# Load the MNIST dataset and split using inbuilt function
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalization + Resize
X_train = X_train / 255.0
X_test  = X_test / 255.0

X_train = X_train.reshape(-1, 28 * 28)
X_test  = X_test.reshape(-1, 28 * 28)

# One Hot Encoding
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)

# Structure Model
model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Set Optimizer and Loss Function

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# Test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

"""
    Result of the model
"""

history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)
plt.show()

"""
    Prediction Testing
"""

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Plot the first 10 test images along with their predicted and true labels
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
