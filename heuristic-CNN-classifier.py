import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

MAX_LENGTH = 24  # Maximum length of input strings
NUM_CLASSES = 15  # Number of possible class labels

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

# Load the data
X, y = load_data('heuristic.json')

# Convert the input data to a 3D tensor for input to a CNN
X = X.reshape((-1, MAX_LENGTH, 1))

# One-hot encode the class labels
y = np.eye(NUM_CLASSES)[y]

# Define the model architecture
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(MAX_LENGTH, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# save model
model.save('cnn_model.h5')
