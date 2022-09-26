import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor

with open('weather2006.csv') as file:
    reader = csv.reader(file, delimiter=',')
    data = np.array(list(reader))

temp = np.array([data[:, 0].astype(float)]).T
hum = np.array([data[:, 1].astype(float)]).T

lof = LocalOutlierFactor()

#model = LinearRegression().fit(hum, temp)   # type: LinearRegression
model = HuberRegressor(epsilon=1.5).fit(hum, temp) # type: HuberRegressor
w0 = model.intercept_
w1 = model.coef_[0]

score = model.score(hum, temp)

# 3-fold Cross Validation
scores = cross_val_score(LinearRegression(), hum, temp, cv=3)

print(f"w0: {w0}\tw1:{w1}\tscore:{score}")
print(scores)
print(np.average(scores))

x = np.array([np.linspace(np.min(hum), np.max(hum), 50)]).T
y = model.predict(x)

plt.scatter(hum, temp, s=3)
plt.plot(x, y, color="red")
plt.xlabel("Humidity")
plt.ylabel("Temperature [C]")
plt.title("Temperature vs. Humidity")
plt.show()
