import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import cross_val_score

# Author: Nathaniel Gatesh
# Date: 26 September 2022

"""
Summary: Two regression types were used, and both gave essentially the same result (see PlotResult.png). The scores for
each were 0.39, which is not particularly good, but understandable given the shape of the original data. The cross
validation showed that both models behaved fairly consistently between each fold. A Huber regressor was tried to limit
the influence of outliers in the dataset, but the result was little change. Learning rates were not adjusted because
both models converged within a few seconds without adjustment. The final impression is that temperature is also
dependent upon other variables besides humidity.
"""

# Import the weather data.
with open('weather.csv') as file:
    reader = csv.reader(file, delimiter=',')
    data = np.array(list(reader))

# Separate data into temperature and humidity.
temp = np.array([data[:, 0].astype(float)]).T
hum = np.array([data[:, 1].astype(float)]).T

# Fit the data using a linear regression.
model1 = LinearRegression().fit(hum, temp)  # type: LinearRegression
model2 = HuberRegressor().fit(hum, temp)    # type: HuberRegressor

# Extract the linear coefficients.
w0_1 = model1.intercept_
w1_1 = model1.coef_[0]
w0_2 = model2.intercept_
w1_2 = model2.coef_[0]

# Get the RMS Error rate for this fit.
score1 = model1.score(hum, temp)
score2 = model2.score(hum, temp)

# 3-fold Cross Validation
scores1 = cross_val_score(LinearRegression(), hum, temp, cv=3)
scores2 = cross_val_score(HuberRegressor(), hum, temp, cv=3)

# Print Statistics
print(f"__Linear Regressor__\n\tw0= {w0_1}\tw1= {w1_1}\tscore= {score1}")
print(f"\tCross Validation Scores: {scores1}")
print(f"\tCV: average= {np.average(scores1)}, stdev= {np.std(scores1)}")

print(f"__Huber Regressor__\n\tw0= {w0_2}\tw1= {w1_2}\tscore= {score2}")
print(f"\tCross Validation Scores: {scores2}")
print(f"\tCV: average= {np.average(scores2)}, stdev= {np.std(scores2)}")

# Generate Prediction
x = np.array([np.linspace(np.min(hum), np.max(hum), 50)]).T
y1 = model1.predict(x)
y2 = model2.predict(x)

# Plot Prediction
plt.scatter(hum, temp, s=3, color='teal')
plt.plot(x, y1, color="red")
plt.plot(x, y2, color="yellow")
plt.xlabel("Humidity")
plt.ylabel("Temperature [C]")
plt.title("Temperature vs. Humidity")
plt.legend(['Weather Data', 'Linear Regressor', 'Huber Regressor'])
plt.show()
