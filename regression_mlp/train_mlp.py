from tkinter.messagebox import NO
from matplotlib.pyplot import sca
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras.models
import keras.layers
from keras.losses import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from models import get_neural_mlp_model, get_sequencial_model


# Obtain the dataset
dataset = fetch_california_housing()
print(dataset.keys())

# Split the dataset between train/validation/test

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.9)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, train_size=0.9)

# Normalize the dataset
scaler = MinMaxScaler()
# The normalization fit must be done only on the train subset
X_train = scaler.fit_transform(X_train)
# Then we transform (normalize) on the rest of the subsets
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)


seq_model = get_sequencial_model()

seq_model.compile(optimizer='adam', loss=mean_squared_error)

# Model training
seq_history = seq_model.fit(x=X_train, y=y_train, batch_size=100,
                            epochs=50, validation_data=(X_valid, y_valid))

# Model evaluation
mse = seq_model.evaluate(x=X_test, y=y_test)

print("Sequential model MSE: {}".format(mse))

# Create non sequential model
no_seq_model = get_neural_mlp_model(X_train.shape[1:])
no_seq_model.compile(optimizer='adam', loss=mean_squared_error)

# Model trainig
no_seq_history = no_seq_model.fit(x=X_train, y=y_train, batch_size=100,
                                  epochs=50, validation_data=(X_valid, y_valid))

# Model evaluation
no_seq_mse = no_seq_model.evaluate(x=X_test, y=y_test)

print("No Sequential model MSE: {}".format(no_seq_mse))


# Plot and save learning curves
pd.DataFrame(seq_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.savefig("seq_learning_history.png")

# Plot and save learning curves
pd.DataFrame(no_seq_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.savefig("no_seq_learning_history.png")
