import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import print
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")

df = pd.read_csv("datasets/gapminder.csv")

# Create arrays for features and target variable
y = df.life.values
X = df.drop(["life", "Region"], axis=1)
X_fertility = df.fertility.values

# Reshape X and y
# from (n rows,) to (n rows, 1)
y = y.reshape(-1, 1)
X_fertility = X_fertility.reshape(-1, 1)

sns.heatmap(df.corr(), square=True, cmap="RdYlGn")

reg = LinearRegression()
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1, 1)
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(f"R-squared for fertility ~ life expectancy; {reg.score(X_fertility, y)}")

# Plot regression line
plt.scatter(X_fertility, y)
plt.plot(prediction_space, y_pred, color="black", linewidth=3)
plt.xlabel("fertility")
plt.ylabel("life expectancy")
plt.show()

# Train/test split for regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print(f"R^2: {reg_all.score(X_test, y_test)}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# using cross-validation
cv_results = cross_val_score(
    reg, X_fertility, y, cv=5
)  # returns an array of cross-validation r^2 scores for each cv
cv_r2 = np.mean(cv_results)


# ridge regression
ridge = Ridge(
    alpha=0.1, normalize=True
)  # ensures all our variables are on the same scale
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# lasso regression
lasso = Lasso(
    alpha=0.1, normalize=True
)  # ensures all our variables are on the same scale
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
