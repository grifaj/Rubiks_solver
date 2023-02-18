import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Define the maximum length of a string (in characters) and the number of possible characters
MAX_LENGTH = 24
NUM_CHARS = 6
NUM_CLASSES = 15

# Load the dataset from a JSON file
def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for sample in data:
            x_i = [int(c) for c in sample]
            X.append(x_i)
            y.append(data[sample])
    return np.array(X), np.array(y)

# Convert the input data to one-hot encoding
def preprocess_data(X, y):
    X = np.eye(NUM_CHARS)[X]
    y = np.eye(NUM_CLASSES)[y]
    return X, y

# Define the RNN model
def build_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(MAX_LENGTH, NUM_CHARS)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load the data from the JSON file
X, y = load_data('heuristic.json')
X, y = preprocess_data(X, y)

# Split the data into training and test sets
num_samples = X.shape[0]
num_train = int(num_samples * 0.8)
X_train, y_train = X[:num_train], y[:num_train]
X_test, y_test = X[num_train:], y[num_train:]

# Train the model on the training dataset
model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

