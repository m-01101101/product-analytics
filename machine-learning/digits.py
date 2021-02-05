from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from rich import print
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

digits = datasets.load_digits()

print(digits.DESCR)
print(digits.keys())

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()  # corresponds to a 5

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

#### comparing classifiers ####
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title("k-NN: Varying Number of Neighbors")
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

print("1 shows overfit, 8 shows underfit. 3 & 5 perform best")
