import numpy as np
from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM
import matplotlib.pyplot as plt

# P3: Self Organizing Map

# Load data
dataset = np.loadtxt("HW4_P1_Iris.csv", delimiter=",", dtype=float)

X = dataset[:, 0:4]
y = dataset[:, 4]

# Separate features for visualization purposes
sepalLength = X[:, 0]
sepalWidth = X[:, 1]
petalLength = X[:, 2]
petalWidth = X[:, 3]

# Fit the Self Organizing Map
som = SOM(m=3, n=1, dim=4)
som.fit(X)
predictions = som.predict(X)

# Plot Results
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
colors = ['red', 'green', 'blue']

ax[0, 0].scatter(sepalLength, petalLength, c=y, cmap=ListedColormap(colors))
ax[0, 0].title.set_text('Actual Classes')
ax[0, 0].set_xlabel('Sepal Length')
ax[0, 0].set_ylabel('Petal Length')

ax[0, 1].scatter(sepalLength, petalLength, c=predictions, cmap=ListedColormap(colors))
ax[0, 1].title.set_text('Predicted Classes')
ax[0, 1].set_xlabel('Sepal Length')
ax[0, 1].set_ylabel('Petal Length')

ax[1, 0].scatter(sepalWidth, petalWidth, c=y, cmap=ListedColormap(colors))
ax[1, 0].set_xlabel('Sepal Width')
ax[1, 0].set_ylabel('Petal Width')

ax[1, 1].scatter(sepalWidth, petalWidth, c=predictions, cmap=ListedColormap(colors))
ax[1, 1].set_xlabel('Sepal Width')
ax[1, 1].set_ylabel('Petal Width')

plt.show()
