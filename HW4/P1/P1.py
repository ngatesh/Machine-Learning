import numpy as np
from time import time
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

dataset = np.loadtxt("HW4_P1_IrisData.csv", delimiter=",", dtype=str)

X = dataset[:, 1:5].astype(float)
y = dataset[:, 5]

# Classify Flowers with Ada Boost
ada = AdaBoostClassifier(n_estimators=50, random_state=7)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=7)

timeStart = time()
n_scores = cross_val_score(ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
timeEnd = time()

print('Ada Boost')
print(f'\t> Mean Accuracy: {np.average(n_scores):.3f} ({np.std(n_scores):.3f})')
print(f'\t> Time: {timeEnd - timeStart:.2f}')

# Classify Flowers with Random Forest
randomForest = RandomForestClassifier(n_estimators=50, random_state=7)
timeStart = time()
n_scores = cross_val_score(randomForest, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
timeEnd = time()

print('Random Forest')
print(f'\t> Mean Accuracy: {np.average(n_scores):.3f} ({np.std(n_scores):.3f})')
print(f'\t> Time: {timeEnd - timeStart:.2f}')

# Classify Flowers with Gradient Boosting
gradientBoost = GradientBoostingClassifier(n_estimators=50, random_state=7)
timeStart = time()
n_scores = cross_val_score(gradientBoost, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
timeEnd = time()

print('Gradient Boost')
print(f'\t> Mean Accuracy: {np.average(n_scores):.3f} ({np.std(n_scores):.3f})')
print(f'\t> Time: {timeEnd - timeStart:.2f}')

# Classify Flowers with Bagging
bagging = BaggingClassifier(n_estimators=50, random_state=7)
timeStart = time()
n_scores = cross_val_score(bagging, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
timeEnd = time()

print('Bagging')
print(f'\t> Mean Accuracy: {np.average(n_scores):.3f} ({np.std(n_scores):.3f})')
print(f'\t> Time: {timeEnd - timeStart:.2f}')

# Classify Flowers with a Single Decision Tree
singleTree = DecisionTreeClassifier(random_state=7)
timeStart = time()
n_scores = cross_val_score(singleTree, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
timeEnd = time()

print('Single Decision Tree')
print(f'\t> Mean Accuracy: {np.average(n_scores):.3f} ({np.std(n_scores):.3f})')
print(f'\t> Time: {timeEnd - timeStart:.2f}')
