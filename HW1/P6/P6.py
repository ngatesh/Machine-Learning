import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Author: Nathaniel Gatesh
# Date: 26 September 2022

"""
Summary: A logistic regression model was trained to predict whether a candy is popular based on its price and sugar
content. The prediction score was 0.68, and 3-fold cross validation showed an average score of 0.66 with standard
deviation 0.043. The prediction rate is okay (better than problem 4) and fairly consistent. See 'PlotResult.png' for
a comparison of actual vs. predicted data. The colors pink and blue were chosen so that overlap would render visually
as a distinct purple color.
"""

# Import the candy data.
with open('candy.csv') as file:
    reader = csv.reader(file, delimiter=',')
    data = np.array(list(reader))

# Extract the sugar and price data, then combine into the independent variable matrix.
sugar = np.array([data[:, 0].astype(float)]).T
price = np.array([data[:, 1].astype(float)]).T
winFactors = np.concatenate((sugar, price), axis=1)

# Separate the win data, then categorize as 1=popular and 2=not-popular
winRate = np.array([data[:, 2].astype(float)]).ravel()
winCat = np.round(np.array([data[:, 2].astype(float)]).ravel() / 100)

# Train the logistic model.
model = LogisticRegression().fit(winFactors, winCat)

# Evaluate the score of the model, with 3-fold cross validation.
score = model.score(winFactors, winCat)
scores = cross_val_score(LogisticRegression(), winFactors, winCat, cv=3)

# Print statistics.
print(f"__Logistic Regressor__")
print(f"\tScore: {score}")
print(f"\tCross Validation Scores: {scores}")
print(f"\tCV: average= {np.average(scores)}, stdev= {np.std(scores)}")

# Plot prediction.
ax = plt.axes(projection='3d')
ax.scatter3D(sugar, price, winCat, color="pink")
ax.scatter3D(sugar, price, model.predict(winFactors), color='blue')
ax.set_xlabel("Sugar Percent")
ax.set_ylabel("Price Percent")
ax.set_zlabel("Win Percent")
ax.legend(["Data", "Prediction"])

plt.show()
