import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

plt.style.use("ggplot")

boston_data = datasets.load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

print(boston_data.DESCR)
#  'CRIM' is per capita crime rate
#  'NX' is nitric oxides concentration
#  'RM' average number of rooms per dwelling
#  The target variable, 'MEDV', is the median value of owner occupied homes in thousands of dollars

X = df.values
y = boston_data.target

# predicting using a single variable
X_rooms = df.RM.values
# -1 means we don't know the number of rows, but we want it compatible with the original array
# 1 is the number of columns we want
# this transformation ensures we have a one dimensional vector with however many rows
# an array of (150,) becomes (150, 1)
X_rooms = X_rooms.reshape(-1, 1)
y = y.reshape(-1, 1)

_ = plt.scatter(X_rooms, y)
_ = plt.ylabel("Value of house /1000 ($)")
_ = plt.xlabel("Number of rooms")
plt.show()

reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(
    -1, 1
)  # create a matrix of values lineary spaced between min and max

plt.scatter(X_rooms, y)
plt.plot(prediction_space, reg.predict(prediction_space), color="black", linewidth=1.5)
plt.ylabel("Value of house /1000 ($)")
plt.xlabel("Number of rooms")
plt.show()

# find the most important features using Lasso regression
lasso = Lasso(alpha=0.1, normalize=True)
lasso_coef = lasso.fit(X, y).coef_

_ = plt.plot(range(len(X.shape[1])), lasso_coef)
_ = plt.xticks(range(len(X.shape[1])), df.columns.values, rotation=60)
_ = plt.ylabel("coefficients")
plt.show()
