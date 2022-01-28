from gc import callbacks
from pyexpat import model
from random import Random
from tkinter.messagebox import NO
from matplotlib.pyplot import sca
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import keras.models
import keras.layers
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor


from models import get_neural_mlp_model_compiled


# Obtain the dataset
dataset = fetch_california_housing()

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


# Param to select the best model
params = {
    # This is needed because is a param of the model creation func, but is a fixed value
    'input_shape': [X_train.shape[1:]],
    'activation_func': ['relu', 'tanh'],
    'neurons': [5, 10, 15, 20, 25, 30]
}

# Wrapper for model selection (Estimator)
model_selection_reg = KerasRegressor(get_neural_mlp_model_compiled)

# Create the random searcher for model selection
random_search = RandomizedSearchCV(
    model_selection_reg, params, n_iter=10, cv=3)

# And run
random_search.fit(X_train, y_train, epochs=50, validation_data=(
    X_valid, y_valid), callbacks=[EarlyStopping(patience=10)])

# Result
print(random_search.best_params_)
