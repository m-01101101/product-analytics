from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

iris = datasets.load_iris()  # loads as a Bunch, similar to a dictionary

X = iris.data  # neat way to drop target: X = df.drop('target', axis=1).values
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

scatter_matrix = pd.plotting.scatter_matrix(
    df, c=y, figsize=[8, 8], s=80, marker="D"
)  # c = color, s=market size

knn = KNeighborsClassifier(n_neighbors=6)

assert (
    iris["data"].shape[0] == iris["target"].shape[0]
)  # same number of observations in features and target
clf = knn.fit(
    iris["data"], iris["target"]
)  # features must be continuous not categorical

X_new = np.array(
    [
        [5.6, 2.8, 3.9, 1.1],
        [5.7, 2.6, 3.8, 1.3],
        [4.7, 3.2, 1.3, 0.3],
    ]
)  # 3 new observations, each with 4 features

assert (
    iris["data"].shape[1] == X_new.shape[1]
)  # same number of features trained on and in new

y_pred =    knn.predict(X_new)

# better approach is to test and split the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=23, stratify=y
)
# stratified sampling aims at splitting a data set so that each split is similar with respect to something
# i.e. the labels are distributed in train and test as they are in the original dataset
# y is the array containing the labels

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)