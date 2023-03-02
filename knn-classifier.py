import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# load your data
labels = ['w', 'r', 'b', 'o', 'g', 'y']
filename = '/home/grifaj/Documents/y3project/Rubiks_solver/colour_data.txt'
#filename = 'C:\\Users\Alfie\\Documents\\uni_work\\year3\\cs310\\github\\Rubiks_solver\\colour_data.txt'
dataset = np.genfromtxt(filename, delimiter=',', dtype=str)
X = [list(map(int, sample)) for sample in dataset[:,:-1]]
y = [labels.index(sample) for sample in dataset[:, -1]]
X = np.array(X)

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

# plot classifier
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

interval = 0.6
xx, yy = np.meshgrid(np.arange(x_min, x_max, interval),
                     np.arange(y_min, y_max, interval))

# predict the class labels for each point in the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# reshape the predicted labels into the meshgrid shape
Z = Z.reshape(xx.shape)

# choose colours
colours = ['white', 'red', 'blue', 'orange', 'green', 'yellow','yellow']

# plot the decision boundaries using a contour plot with custom colors
plt.contourf(xx, yy, Z, colors=colours, alpha=0.4)

# plot the scatter points with custom colors
for i in range(6):
    idx = [p for p in range(len(y)) if y[p] == i]
    plt.scatter(X[idx, 0], X[idx, 1], c=colours[i], s=20, edgecolor='k')

plt.show()

# Save the classifier to disk
joblib.dump(knn, 'knn.joblib')


