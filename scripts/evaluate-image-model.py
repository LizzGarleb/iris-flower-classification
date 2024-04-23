from keras.metrics import Precision, Recall, BinaryAccuracy
from modules.imageModel.imageModel import ImageModel
from modules.imageModel.dataHandler import DataHandler

"""
Evaluate the ImageModel using test data and calculate precision, recall, and accuracy metrics.

Imports:
    - Precision: Keras metric for precision calculation.
    - Recall: Keras metric for recall calculation.
    - BinaryAccuracy: Keras metric for binary accuracy calculation.
    - ImageModel: Custom class for the image classification model.
    - DataHandler: Custom class for handling image data loading and preprocessing.

Steps:
    1. Initialize the ImageModel.
    2. Load the trained model from the ImageModel instance.
    3. Initialize precision, recall, and accuracy metrics.
    4. Initialize DataHandler to load and preprocess the test data.
    5. Split the test data into batches.
    6. Iterate through the test data batches:
        - Make predictions using the model.
        - Update the metrics with the true labels and predicted values.
    7. Print the calculated precision, recall, and accuracy metrics.

Note:
    - The test data is assumed to be in the 'data/images' directory.
    - The ImageModel and DataHandler classes are custom implementations for image classification tasks.
"""

image_model = ImageModel()
model = image_model.model

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

data_handler = DataHandler(directory='data/images')
data = data_handler.preprocess_data()
train, val, test = data_handler.split_data()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result()}, Recal: {re.result()}, Accuracy: {acc.result()}')
