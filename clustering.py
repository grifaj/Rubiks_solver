import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# load your data
labels = ['o', 'g', 'b', 'y', 'w', 'r'] #labels = ['w', 'r', 'b', 'o', 'g', 'y']
#filename = '/home/grifaj/Documents/y3project/Rubiks_solver/colour_data.txt'
filename = 'C:\\Users\Alfie\\Documents\\uni_work\\year3\\cs310\\github\\Rubiks_solver\\colour_data.txt'
dataset = np.genfromtxt(filename, delimiter=',', dtype=str)
X = [list(map(int, sample)) for sample in dataset[:,:-1]]
y = [labels.index(sample) for sample in dataset[:, -1]]
X = np.array(X)

np.random.seed(42)

# Instantiate a KMeans object with the number of clusters you want
kmeans = KMeans(n_clusters=6)

# Fit the KMeans object to the data
kmeans.fit(X)

# Get the labels assigned to each data point
labels = kmeans.labels_

# Get the centroids of each cluster
centroids = kmeans.cluster_centers_

# calculate accuracy
accuracy = accuracy_score(labels, y)
print("Accuracy:", accuracy)

# plot classifier
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

interval = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, interval),
                     np.arange(y_min, y_max, interval))

# predict the class labels for each point in the meshgrid
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# reshape the predicted labels into the meshgrid shape
Z = Z.reshape(xx.shape)

# choose colours
colours = ['orange', 'green', 'blue', 'yellow', 'white', 'red','red']

# plot the decision boundaries using a contour plot with custom colors
plt.contourf(xx, yy, Z, colors=colours, alpha=0.4)

# choose colours
colours = ['orange', 'green', 'blue', 'yellow', 'white', 'red']
# plot the scatter points with custom colors
for i in range(6):
    idx = [p for p in range(len(labels)) if labels[p] == i]
    plt.scatter(X[idx, 0], X[idx, 1], c=colours[i], s=20, edgecolor='k')

# plot centroids 
plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='k')

plt.title('clustering on sample data')
plt.show()

