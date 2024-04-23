import pandas as pd
from tkinter import Tk, filedialog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class TextModel:
    """
    A class to represent a TextModel using K-Nearest Neighbors classifier for iris flower classification.

    Attributes:
        n_neighbors (int): Number of neighbors for the K-Nearest Neighbors classifier.
        knn (KNeighborsClassifier): K-Nearest Neighbors classifier instance.
        iris (Bunch): Dictionary-like object holding iris dataset.
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Testing features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Testing labels.
        x_new (list): Input features for prediction.

    Methods:
        _load_data(self): Load and preprocess the iris dataset.
        _train_model(self): Train the K-Nearest Neighbors classifier.
        predict(self, x_new=None): Make predictions based on the input data.
        csv_input(self): Load input data from a CSV file.
        test_accuracy(self): Test the accuracy of the trained model.
    """
    def __init__(self, n_neighbors=1):
        """
        Initialize the TextModel class with a K-Nearest Neighbors classifier.

        Args:
            n_neighbors (int, optional): Number of neighbors for the K-Nearest Neighbors classifier. Defaults to 1.
        """
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self._load_data()
        self._train_model()

    def _load_data(self):
        """
        Load and preprocess the iris dataset.
        """
        self.iris = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.iris['data'], self.iris['target'], random_state=0)

    def _train_model(self):
        """
        Train the K-Nearest Neighbors classifier.
        """
        self.knn.fit(self.X_train, self.y_train)

    def predict(self, x_new=None):
        """
        Make predictions based on the input data.

        Args:
            x_new (list, optional): List of input features for predictions.

        Returns:
            list: Predicted iris species names.
        """
        if x_new is not None:
            self.x_new = x_new

        flower_names = []
        for x in x_new:
            prediction = self.knn.predict([x])
            flower_name = self.iris['target_names'][prediction]
            flower_names.append(flower_name[0])
        return flower_names

    def csv_input(self):
        """
        Load input data from a CSV file and return it as a list of values.

        Returns:
            list: List of input values loaded from the CSV file.
        """
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        data = pd.read_csv(file_path)
        return data.values.tolist()

    def test_accuracy(self):
        """
        Test the accuracy of the trained model.

        Returns:
            float: Accuracy score of the model.
        """
        return self.knn.score(self.X_test, self.y_test)
