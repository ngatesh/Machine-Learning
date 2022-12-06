import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[168, 156, 176, 256, 230, 116, 242, 242, 174, 1004, 1228, 964, 2008],
              [3, 3, 3.5, 3, 5, 3, 7, 4.5, 2.5, 35, 46, 17, 32],
              [1814, 1358, 2200, 2070, 1410, 1238, 1315, 1183, 1110, 1218, 1889, 2120, 1257],
              [15, 14, 16, 27, 131, 104, 104, 78, 73, 81, 82.4, 20, 13],
              [0.001, 0.01, 0.005, 0.2, 3.5, 0.06, 0.01, 0.02, 1.5, 1172, 1932, 1030, 1038],
              [1879, 1425, 2140, 2700, 1575, 1221, 1434, 1374, 1256, 33.3, 43.1, 1966, 1289]], dtype=float).T

y = np.array(["ok", "ok", "ok", "ok", "settler", "settler", "settler", "settler", "settler", "solids", "solids", "solids", "solids"]).T

gnb = GaussianNB()
gnb.fit(X, y)

prediction = gnb.predict(np.array([[222, 4.5, 1518, 74, 0.25, 1642]]))
print(prediction)

# Output: ['settler']
