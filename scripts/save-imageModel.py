from modules.imageModel.imageModel import ImageModel
from modules.imageModel.dataHandler import DataHandler

"""
Script for training an image classification model and saving the trained model and training history.

This script utilizes an ImageModel class and a DataHandler class to preprocess the image data, 
train an image classification model, and save both the trained model and its training history.

Dependencies:
    - ImageModel and DataHandler classes (imported from modules.imageModel.imageModel and modules.imageModel.dataHandler)
    
Usage:
    - Ensure that the image data directory ('data/images') contains the training images organized in subdirectories by class.
    - Run the script to preprocess the data, train the model, and save the trained model and training history.
    
Example:
    python train_and_save_model.py

Note:
    - This script assumes that the ImageModel class and the DataHandler class are correctly implemented 
      and available in the specified modules.
    - The trained model and its training history are saved to the specified file paths.
"""

data_handler = DataHandler(directory='data/images')

train_data, val_data, test_data = data_handler.split_data()

image_model = ImageModel()

history = image_model.compile_and_train(train_data, val_data)

model_path = 'models/image_model.h5'

image_model.save_model_and_history(history, model_path=model_path)
print(f"Model and history saved to {model_path}")

image_model.plot_accuracy(history)
image_model.plot_losses(history)
