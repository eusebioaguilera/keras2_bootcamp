import keras.models
import keras.layers


def get_sequencial_model():
    """
        This method builds a sequential model based on MLP to solve the regression problem
    """
    model = keras.models.Sequential(
        [
            # Hidden layer with 30 neurons and the relu activation function
            keras.layers.Dense(30, activation="relu"),
            # Output layer with one neuron (the value predicted) and no activation function
            keras.layers.Dense(1)
        ]
    )

    return model


def get_neural_mlp_model(input_shape, activation_func='relu', neurons=30):
    """
        This method implements a model with a non-sequential topology. For this purpose we need to
        use the Functional API provided by Keras. 
        The topology simply concatenate the input to the final hidden layer in order to add information in the final step
    """
    input = keras.layers.Input(shape=input_shape)
    hidden_layer_1 = keras.layers.Dense(
        units=neurons, activation=activation_func)(input)
    hidden_layer_2 = keras.layers.Dense(
        units=neurons, activation=activation_func)(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(
        units=neurons, activation=activation_func)(hidden_layer_2)
    concat_layer = keras.layers.Concatenate()([input, hidden_layer_3])
    output_layer = keras.layers.Dense(1)(concat_layer)
    # Create the model defining the input layers and the output ones
    model = keras.models.Model(inputs=[input], outputs=[output_layer])

    return model
