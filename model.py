from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset and assign to a variable.
iris = load_iris()

# Transpose the data to get the features as columns and samples as rows.
features = iris.data.T

# Data Seperated
sepal_lenght = features[0]
sepal_width = features[1]
petal_lenght = features[2]
petal_width = features[3]

# Label Seperated
sepal_lenght_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_lenght_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]


# Split data in two set. One to train and another to test.
X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0)

# K-Nearest Neighboard Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Test accuracy of the model.
# print(knn.score(X_test, y_test))


