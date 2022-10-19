import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

# Author: Nathaniel Gatesh
# Date: 19 October 2022

"""
Result: For a linear regression problem, hidden layers are unnecessary, because a single neuron with a bias can model
a linear relationship between its input and output. When testing the effect of additional hidden layers of various
widths and depths, the final loss values for the trained networks remained almost the same. Although the loss values
remained the same, each model did produce a distinct line on the results graph. This could be caused by a small loss
gradient around the solution.
"""

# Read data from .csv file.
dataset = np.loadtxt("weather.csv", delimiter=",")
temp = dataset[:, 0].astype(float)
hum = dataset[:, 1].astype(float)

# Initialize optimizer
sgd = SGD(learning_rate=0.01)

# No hidden layers.
print("\n0x0:")
model0 = Sequential()
model0.add(Dense(1, use_bias='true', input_dim=1))

model0.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h0 = model0.fit(hum, temp, epochs=2)

# One Hidden Layer, One Neuron Each
print("\n1x1:")
model1 = Sequential()
model1.add(Dense(1, use_bias='true', input_dim=1))
model1.add(Dense(1, use_bias='true'))

model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h1 = model1.fit(hum, temp, epochs=2)

# One Hidden Layer, Two Neurons Each
print("\n1x2:")
model2 = Sequential()
model2.add(Dense(2, use_bias='true', input_dim=1))
model2.add(Dense(1, use_bias='true'))

model2.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h2 = model2.fit(hum, temp, epochs=2)

# One Hidden Layer, Three Neurons Each
print("\n1x3:")
model3 = Sequential()
model3.add(Dense(3, use_bias='true', input_dim=1))
model3.add(Dense(1, use_bias='true'))

model3.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h3 = model3.fit(hum, temp, epochs=2)

# Two Hidden Layers, One Neuron Each
print("\n2x1:")
model4 = Sequential()
model4.add(Dense(1, use_bias='true', input_dim=1))
model4.add(Dense(1, use_bias='true'))
model4.add(Dense(1, use_bias='true'))

model4.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h4 = model4.fit(hum, temp, epochs=2)

# Two Hidden Layers, Two Neurons Each
print("\n2x2:")
model5 = Sequential()
model5.add(Dense(2, use_bias='true', input_dim=1))
model5.add(Dense(2, use_bias='true'))
model5.add(Dense(1, use_bias='true'))

model5.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h5 = model5.fit(hum, temp, epochs=2)

# Two Hidden Layers, Three Neurons Each
print("\n2x3:")
model6 = Sequential()
model6.add(Dense(3, use_bias='true', input_dim=1))
model6.add(Dense(3, use_bias='true'))
model6.add(Dense(1, use_bias='true'))

model6.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
h6 = model6.fit(hum, temp, epochs=2)

# Plot Results
x = np.array([np.linspace(np.min(hum), np.max(hum), 10)]).T

plt.scatter(hum, temp, s=3, color='teal')
legends = ["Weather Data"]

plt.plot(x, model0.predict(x))
legends.append(f"Zero: {h0.history['mean_squared_error'][-1]:.2f}")

plt.plot(x, model1.predict(x))
legends.append(f"1x1: {h1.history['mean_squared_error'][-1]:.2f}")

plt.plot(x, model2.predict(x))
legends.append(f"1x2: {h2.history['mean_squared_error'][-1]:.2f}")

plt.plot(x, model3.predict(x))
legends.append(f"1x3: {h3.history['mean_squared_error'][-1]:.2f}")

plt.plot(x, model4.predict(x))
legends.append(f"2x1: {h4.history['mean_squared_error'][-1]:.2f}")

plt.plot(x, model5.predict(x))
legends.append(f"2x2: {h5.history['mean_squared_error'][-1]:.2f}")

plt.plot(x, model6.predict(x))
legends.append(f"2x3: {h6.history['mean_squared_error'][-1]:.2f}")

plt.xlabel("Humidity")
plt.ylabel("Temperature [C]")
plt.title("Temperature vs. Humidity\nHidden Layers: Loss")
plt.legend(legends)
plt.show()
