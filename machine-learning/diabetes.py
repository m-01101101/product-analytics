"""
The PIMA Indians dataset obtained from the UCI Machine Learning Repository
The goal is to predict whether or not a given female patient will contract diabetes 
    based on features such as BMI, age, and number of pregnancies
It is a binary classification problem    
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

plt.style.use("ggplot")

_df = pd.read_csv("datasets/diabetes.csv")
df = _df.dropna()

X = df.drop("Outcome", axis=1).values
# X = X.reshape(-1, 8)

y = df.Outcome.values
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_test, y_test)
y_pred = knn.predict(X_test)

print("k-NN performance")
# must always be (test, prediction)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#  the support columns gives the number of samples of the true response that lie in that class

#### logistic regression ####

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("logistic regression performance")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# predict_proba returns an array with two columns: each column contains the probabilities for the respective target values.
#  we choose the second column, the one with index 1,
# that is, the probabilities of the predicted labels being '1'
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

_ = plt.plot([0, 1], [0, 1], "k--")
_ = plt.plot(fpr, tpr)
_ = plt.xlabel("False Positive Rate")
_ = plt.ylabel("True Positive Rate")
_ = plt.title("ROC Curve")
plt.show()

print(f"AUC: {roc_auc_score(y_test, y_pred_prob)}")

cv_auc = cross_val_score(logreg, X, y, cv=5, scoring="roc_auc")

#### hyperparameter tuning ####
# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {"C": c_space}  # hyperparameter to tune and values to test
logreg = LogisticRegression()
logreg_cv = GridSearchCV(
    logreg, param_grid, cv=5
)  # instantiate the GridSearchCV object
logreg_cv.fit(X, y)  # fits in place

print(
    f"""Tuned Logistic Regression Parameters: {logreg_cv.best_params_}
Best score is {logreg_cv.best_score_}"""
)

#### random tuning ####
tree = DecisionTreeClassifier()

param_dist = {
    "max_depth": [3, None],
    "max_features": randint(1, 9),
    "min_samples_leaf": randint(1, 9),
    "criterion": ["gini", "entropy"],
}

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X, y)

print(
    f"""Tuned Decision Tree Parameters: {tree_cv.best_params_}
Best score is {tree_cv.best_score_}"""
)
