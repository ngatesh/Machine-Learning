import numpy as np

# Numbers 3-10, excluding 10, spaced by 1.
W = np.array([np.arange(3, 10, 1)])
# Numbers 3-10, including 10, five total numbers.
X = np.array([np.linspace(3, 10, 5)])
# 3x3 array of ones.
Y = np.ones([3, 3])
# 2x4 array of zeros.
Z = np.zeros([2, 3])

J = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 8, 1))

X = np.power(np.array([1, 2, 3, 4]), 2)

print(X)

