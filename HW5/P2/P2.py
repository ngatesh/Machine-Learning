import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# P2: K-Means Clustering

# Load Data
X = np.loadtxt("Live_20210128.csv", delimiter=",", dtype=int)

# Different numbers of clusters to try
k = range(1, 11)
variation = np.zeros(10)

for i in k:
    # Regressor model
    kmeans = KMeans(n_clusters=i, random_state=7).fit(X)
    # Record variation
    variation[i-1] = kmeans.inertia_

# Plot results
plt.scatter(k, variation)
plt.title('K-Means Clustering: Variation vs. Cluster Count')
plt.xlabel('Clusters')
plt.ylabel('Variation')
plt.show()
