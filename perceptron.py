import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# load your data
labels = ['w', 'r', 'b', 'o', 'g', 'y']
filename = '/home/grifaj/Documents/y3project/Rubiks_solver/colour_data.txt'
dataset = np.genfromtxt(filename, delimiter=',', dtype=str)
X = [list(map(int, sample)) for sample in dataset[:,:-1]]
y = [labels.index(sample) for sample in dataset[:, -1]]


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# fit the classifier to the training data
knn.fit(X_train, y_train)

# make predictions on the testing data
y_pred = knn.predict(X_test)

# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the classifier to disk
joblib.dump(knn, 'knn.joblib')


