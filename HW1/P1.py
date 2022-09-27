import numpy as np
import matplotlib.pyplot as plt

# Plot the sigmoid function and its derivative.
x = np.linspace(-25, 25, 101)
S = 1 / (1 + np.exp(-x))
dSdx = S * (1 - S)

plt.plot(x, S)
plt.plot(x, dSdx)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Sigmoid", "Derivative"])
plt.title("The Sigmoid and its Derivative")

plt.show()
