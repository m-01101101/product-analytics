import numpy as np
import pandas as pd
from rich import print
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("datasets/winequality-white.csv", delimiter=";")

y = df.quality.values
y.reshape(-1, 1)
X = df.drop("quality", axis=1).values
X_scaled = scale(X)

print(
    f"""Mean of Unscaled Features\n{np.mean(X)}
Standard Deviation of Unscaled Features:\n{np.std(X)}

Mean of Scaled Features:\n{np.mean(X_scaled)}
Standard Deviation of Scaled Features:\n{np.std(X_scaled)}
"""
)

steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn_scaled = pipeline.fit(X_train, y_train)
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

print(
    f"""Accuracy with Scaling: {knn_scaled.score(X_test, y_test)}
Accuracy without Scaling: {knn_unscaled.score(X_test, y_test)}
"""
)
