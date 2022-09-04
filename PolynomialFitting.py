import numpy as np
import matplotlib.pyplot as plt

# # Generate Training Data # #

# Initialize x-space (Nx2 matrix).

X0 = np.ones([20, 1])
X1 = np.array([np.arange(1, 21, 1)]).T
X2 = np.power(np.array([np.arange(1, 21, 1)]), 2).T
X = np.concatenate((X0, X1, X2), axis=1)

# Define W0 = 20, W1 = 3 (2x1 matrix).
W_real = np.array([[20, 5, 0]]).T

# Define Y = W0 + W1*x + [noise]
Y = X.dot(W_real) + np.random.rand(20, 1)*40 - 20

# # Optimization # #

W = np.array([[19, 5, 0]]).T     # Initial guess: W0 = 0, W1 = 0;
a = 0.000002                      # Learning Rate
ll = 0                           # Shrinkage coefficient lambda
LGrad = np.array([[1, 1, 1]]).T  # Loss function gradient w.r.t. W=[W0, W1]

count = 0

while np.abs(LGrad[0]) > pow(10, -2) or np.abs(LGrad[1]) > pow(10, -2) or np.abs(LGrad[2]) > pow(10, -2):
    h = X.dot(W)                                        # Prediction
    lq = np.sum(W)                                      # Calculate lasso contour value
    LGrad = (Y-h).T.dot(X).T - ll * lq/np.abs(lq)       # Calculate Gradient
    W = W + a * LGrad                                   # Update W0, W1

    count = count + 1
    if count % 10000 == 0:
        print(W.T)
        print("\t", end="")
        print(LGrad.T)

print(W.T)
print(LGrad.T)

plt.scatter(X1, Y)
plt.plot(X1, X.dot(W))

plt.show()