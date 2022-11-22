import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# P4: Gaussian Mixture Model

# Load data
X = np.loadtxt("HW5_P4_CC GENERAL.csv", delimiter=",", dtype=float)

# Different numbers of components to try
n = range(1, 20)
scores = np.zeros(19)

# Score each model
for i in n:
    gm = GaussianMixture(n_components=i, random_state=7).fit(X)
    scores[i-1] = gm.bic(X)
    print(f'n={i}\tScore: {scores[i-1]:.3f}')

# Plot results
plt.scatter(n, scores)
plt.title('Gaussian Mixture Model')
plt.xlabel('N-Components')
plt.ylabel('BIC Score')
plt.show()

