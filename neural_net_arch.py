import tensorflow as tf
from tensorflow.keras import layers, models

# Example 1: Multilayer Perceptron (MLP)
def create_mlp(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Example 2: Convolutional Neural Network (CNN)
def create_cnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Example 3: Recurrent Neural Network (RNN) - LSTM
def create_rnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Example usage
input_shape_mlp = (784,)  # Example for MLP with 28x28 images
input_shape_cnn = (32, 32, 3)  # Example for CNN with 32x32 RGB images
input_shape_rnn = (10, 1)  # Example for RNN with sequence length 10

num_classes = 10  # Example: 10 classes for classification

mlp_model = create_mlp(input_shape_mlp, num_classes)
cnn_model = create_cnn(input_shape_cnn, num_classes)
rnn_model = create_rnn(input_shape_rnn, num_classes)

mlp_model.summary()
cnn_model.summary()
rnn_model.summary()
