import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# P1: K-Nearest-Neighbor

# Load Data
dataset = np.loadtxt("seeds_dataset.txt", delimiter="\t", dtype=float)

# Separate
X = dataset[:, 0:7]
y = dataset[:, 7].astype(int)

k = range(1, 50)
errors = np.zeros(49)

for i in k:
    # Cross validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=7)
    # Classifier model
    knn = KNeighborsClassifier(n_neighbors=i)

    # Score the model with cross validation
    n_scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    # Print results
    print(f"K-Nearest-Neighbor (k={i})")
    print(f'\t> Mean Accuracy (stdev): {np.average(n_scores):.3f} ({np.std(n_scores):.3f})')

    errors[i-1] = np.average(n_scores)

plt.scatter(k, errors)
plt.title("KNN - Accuracy vs. Number of Neighbors")
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.show()
