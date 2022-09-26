import numpy as np
from matplotlib import pyplot as plt

x0 = np.ones([1, 100])
x1 = np.array([np.arange(1, 21, 0.2)])

X = np.concatenate((x0, x1), axis=0)
W = np.array([[20, 5]])
Y = W.dot(X) + np.random.rand(1, 100)*40 - 20

lGrad = np.array([[1, 1, 1]])
lGradLim = np.array([[1, 1, 1]]) * 10**-1
W_guess = np.array([[19, 5, 0]])
a = 0.00002

count = 0

while np.greater(np.abs(lGrad), lGradLim).any():
    h = W_guess.dot(X)
    lGrad = (Y - h).dot(X.T)
    W_guess = W_guess + a * lGrad

    count = count+1
    if count % 10000 == 0:
        print(W_guess)
        print(f"\t{lGrad}")

print("Final Answer: ")
print(W_guess)
print(f"\t{lGrad}")

plt.scatter(x1.T, Y.T)
plt.plot(x1.T, W_guess.dot(X).T)

plt.show()
