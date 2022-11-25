import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# P6: Support Vector Machine

# Generate random clusters
cluster1 = np.random.normal(loc=-5, scale=3, size=(200, 2))
cluster2 = np.random.normal(loc=8, scale=3, size=(200, 2))

X = np.concatenate([cluster1, cluster2], axis=0)

# Populate category indicators (0/1)
y1 = np.zeros(200)
y2 = np.ones(200)
Y = np.concatenate([y1, y2])

# Train SVM model
svm = LinearSVC().fit(X, Y)

# Get info for plotting decision boundary
w = svm.coef_[0]
b = svm.intercept_[0]
x = np.array([-15, 15])
y = -(w[0] / w[1]) * x - b / w[1]

# Plot results
plt.scatter(X[:, 0], X[:, 1])
plt.plot(x, y, c='r')
plt.title("Linear Support Vector Machine")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
