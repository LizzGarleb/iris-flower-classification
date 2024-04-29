from keras.metrics import Precision, Recall, BinaryAccuracy
from modules.imageModel.imageModel import ImageModel
from modules.imageModel.dataHandler import DataHandler

"""
Script for evaluating a trained image classification model using precision, recall, and binary accuracy metrics.

This script imports a pre-trained image classification model and evaluates its performance using precision, recall, 
and binary accuracy metrics on a test dataset. It uses TensorFlow/Keras metrics for evaluation.

Dependencies:
    - TensorFlow (imported as tf)
    - Keras metrics: Precision, Recall, BinaryAccuracy (imported from keras.metrics)

Usage:
    - Ensure that the trained model and the test dataset are available.
    - Update the paths to the model and dataset if necessary.
    - Run the script to evaluate the model's performance.

Example:
    python evaluate_image_model.py

Note:
    - This script assumes that the model is pre-trained and available for evaluation.
    - It assumes that the test dataset has been preprocessed and is ready for evaluation.
    - The script evaluates the model's performance on the entire test dataset.
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
