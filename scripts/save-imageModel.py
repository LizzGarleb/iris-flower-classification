from modules.imageModel.imageModel import ImageModel
from modules.imageModel.dataHandler import DataHandler

"""
This script demonstrates the process of training an image classification model using the ImageModel class.
It initializes a DataHandler to load and split the data, then trains the ImageModel using the loaded data.
The trained model is saved along with its training history, and the accuracy and loss plots are displayed.

Steps:
    1. Initialize a DataHandler to manage image data.
    2. Split the data into training, validation, and test sets.
    3. Initialize an ImageModel.
    4. Compile and train the ImageModel using the training and validation data.
    5. Save the trained model and its history to a specified path.
    6. Display the accuracy and loss plots based on the training history.
"""

# Initialize DataHandler
data_handler = DataHandler(directory='data/images')

train_data, val_data, test_data = data_handler.split_data()

image_model = ImageModel()

history = image_model.compile_and_train(train_data, val_data)

model_path = 'models/image_model.h5'

image_model.save_model_and_history(history, model_path=model_path)
print(f"Model and history saved to {model_path}")

image_model.plot_accuracy(history)
image_model.plot_losses(history)
