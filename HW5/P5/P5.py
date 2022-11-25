import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# P5: Hierarchical Clustering Model

# Load data
X = np.loadtxt("HW5_P4_CC GENERAL.csv", delimiter=",", dtype=float)

# Fit model
hClustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
hClustering.fit(X)


# Code snippet from official scikit-learn:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# Plot Results
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(hClustering, truncate_mode="level", p=3)    # Change p=1,2,3,4 to visualize different depths.
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
