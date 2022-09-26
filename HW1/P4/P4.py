import numpy as np
import random
import matplotlib.pyplot as plt

# Author: Nathaniel H. Gatesh
# Date: 25 September 2022
"""
Summary: After some experimentation, a learning rate of alpha=0.01 was found to work well; larger values resulted in
overshooting. Without the use of regularization techniques, over fitting occurred after the 50th order polynomial fit
(see noReg.png).

With lasso regularization (ll=0.003), over fitting was improved to the 65th order polynomial (see LassoReg.png).
With ridge regularization (ll=0.003), over fitting was improved to the 70th order polynomial (see RidgeReg.png).

"""

# Generate curve to fit using cos(2*pi*x) from x = 0 to 1.
x = np.array([np.linspace(0.01, 1, 100)])
y = np.cos(2*np.pi*x)

# Add random noise to the curve.
np.random.seed(777)
z = y + (np.random.rand(1, 100) - 0.5)

# Randomize the order of the data points
random.seed(777)
randIndices = random.sample(range(0, 100), 100)

# Split the data points into 80/20 training/testing data.
z_train = z[:, randIndices[0:80]]
z_test  = z[:, randIndices[80:100]]
x_train = x[:, randIndices[0:80]]
x_test  = x[:, randIndices[80:100]]


# Helper function that takes the x-space for the curve and creates a 'deg'-order polynomial version of that x-space,
# and a corresponding 1x(deg+1) w matrix initialized to zero.
# E.g.:                           [ 1        1       1       ]
#       X = [x1 x2 x3] --> xGen = | x1       x2       x3     |
#                                 |(x1)^2   (x2)^2   (x3)^2  |
#                                 | ...      ...      ...    |
#                                 [(x1)^deg (x2)^deg (x3)^deg]
#
#                      --> wGen = [ 0 0 0 0 ]
def polyGen(x, deg):
    wGen = np.zeros((1, deg+1))
    xGen = x**0

    for i in range(1, deg+1):
        xGen = np.concatenate((xGen, x**i), axis=0)

    return xGen, wGen


# Fits the generated curve+noise to a polynomial of the given degree.
# reg = regression type ('lasso' and 'ridge' supported, default = 'none').
# ll = regression coefficient (default = 1).
def polyFit(deg, reg='none', ll=0.0001):
    (X, w) = polyGen(x_train, deg)               # initialize polynomial x, and w.
    alpha = 0.01                                 # Learning rate.

    grad = 1                                     # Initial gradient value so that 'while' will run.
    gradLim = np.ones((1, deg+1)) * 5*10**-3     # Minimum gradient before loop stops learning.

    while np.greater(abs(grad), gradLim).any():
        grad = -(z_train - w.dot(X)).dot(X.T)    # Calculate d(L2)/dw

        if reg == 'lasso':
            grad = grad + ll*np.sign(w)
        elif reg == 'ridge':
            grad = grad + 2*ll*w

        w = w - alpha*grad                       # Update w

    # plt.scatter(x_train, z_train)
    # plt.scatter(x_train, w.dot(X))
    # plt.title(f"M={deg}")
    # plt.show()

    # Calculate RMS error for training set.
    errorTrain = z_train - w.dot(X)
    rmsErrorTrain = np.sqrt(np.sum(errorTrain**2) / np.size(errorTrain))

    # Calculate RMS error for testing set.
    X_test = polyGen(x_test, deg)[0]
    errorTest = z_test - w.dot(X_test)
    rmsErrorTest = np.sqrt(np.sum(errorTest**2) / np.size(errorTest))

    # Print Results.
    print(f"M={deg}\tTrain Error: {rmsErrorTrain},\tTest Error: {rmsErrorTest}")

    return rmsErrorTrain, rmsErrorTest


# Generate list of polynomial orders to fit.
M = np.concatenate(([1, 2, 3, 4], np.arange(5, 81, 5)), axis=0)
length = np.size(M)

# Allocate space to store training and testing errors for each fit.
errTrain = np.zeros((1, length))
errTest = np.zeros((1, length))

# Fit all the polynomials.
for i in range(0, length):
    (errTrain[0, i], errTest[0, i]) = polyFit(M[i], reg='none', ll=0.003)

# Show training and testing errors as polynomial order increases.
plt.scatter(M, errTrain)
plt.scatter(M, errTest)
plt.xlabel("Polynomial Order")
plt.ylabel("RMS Error")
plt.title("RMS Error vs. Polynomial Order")
plt.legend(["Training", "Testing"])
plt.show()


