import multiprocessing
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt

# In this example we are goint to use the MNIST-fashion dataset in order to build
# an image classifier using an MLP based model.

# This method create the model used for training


def create_model(input_shape, layers):
    # We create a Sequential model
    model = keras.models.Sequential()
    # Out first layer is the input one, which is a Flatten input array
    model.add(keras.layers.Flatten(input_shape=input_shape))

    for layer in layers:
        model.add(keras.layers.Dense(
            layer['units'], activation=layer['activation']))

    return model


# Get the dataset from keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# We must split the dataset into train / validate sets. We also must normalize the
# data input.
idx = 5000
X_valid, X_train = X_train_full[:idx] / 255.0, X_train_full[idx:] / 255.0
y_valid, y_train = y_train_full[:idx], y_train_full[idx:]

# Theese are the classes used in the dataset
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Create the model, that is a modelo with an input layer (28x28),
# two hidden layers with 100 and 300 units/neurons recpectively and relu as the activation function.
# And the output is a softmax function for 10 classes
model = create_model(input_shape=[28, 28], layers=[
    {'units': 100, 'activation': 'relu'},
    {'units': 300, 'activation': 'relu'},
    {'units': 10, 'activation': 'softmax'}
])

# We compile the model, set loss function, optimizer and metrics
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Finally, run the training for a number of epochs
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid))

# Plot and save learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.savefig("learaning_history.png")
