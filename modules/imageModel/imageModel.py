import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential, save_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tkinter import filedialog, Tk
import os

class ImageModel:
    """
    A class for building, training, and evaluating a convolutional neural network (CNN)
    model for image classification tasks using TensorFlow and Keras.

    Attributes:
        model (tf.keras.Sequential): CNN model architecture for image classfication.
        history (tf.keras.callbacks.History): Training history containing loss and accuracy metrics.

    Methods:
        _create_model(input_shape): Create and return the CNN model architecture.
        compile_and_train(train_data, val_data, logdir='logs'): Compile and train the CNN model
          using the provided training and validation datasets.
        img_input(self): Loads an image from the user's computer and returns the path.
        save_model_and_history(model_path='models/image_model.h5'): Save
          the trained model, accuracy and loss plots, and model summary to the specified
          paths.
        load_and_predict(image_path, model_path='models/image_model.h5'): Load the trained model from the
          specified path and predict the class of the provided image.
        summary(): Display a summary of the CNN model architecture.
        plot_metric(history, metric): Plot and display a specific metric from the training history.
        plot_loss(history): Plot and display the loss metric from the training history.
        plot_accuracy(history): Plot and display the accuracy metric from the training history.
    """

    def __init__(self, input_shape=(256, 256, 3)):
        """
        Initialize the ImageModel class with a given input shape.

        Args:
            input_shape (tuple, optional): Inputs of the image data (height, width, channels).
            Defaults to (256, 256, 3).
        """
        self.model = self._create_model(input_shape)

    def _create_model(self, input_shape):
        """
        Create & return the image classification model.

        Args:
            input_shape (tuple): Input shape of the image data (height, width, channels).

        Returns:
            tf.keras.Sequential: Compiles image classification model.
        """
        model = Sequential([
            Conv2D(16, (3, 3), 1, activation='relu', input_shape=input_shape),
            MaxPooling2D(),
            Conv2D(32, (3, 3), 1, activation='relu'),
            MaxPooling2D(),
            Conv2D(16, (3, 3), 1, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def compile_and_train(self, train_data, val_data, logdir='logs'):
        """
        Compile and train the image classification model.

        Args:
            train_data (tf.data.Dataset): Training dataset.
            val_data (tf.data.Dataset): Validation dataset.
            logdir (str, optional): Directory path for TensorBoard logs. Defaults to 'logs'.

        Returns:
            tf.keras.callback.History: Training history.
        """
        self.model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = self.model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])
        return hist

    def save_model_and_history(self, history, model_path='models'):
        """
        Save the model, and plots to specified paths.
        
        Parameters:
        - model_path (str): File path to save the model (default is 'models/image_model.keras').
        """

        # Save the model
        self.model.save(os.path.join('model_path', 'image_model.h5'))
        print(f"Model saved to {model_path}")

        # Plot and save accuracy and loss
        self.plot_metric(history, 'accuracy')
        plt.savefig('models/accuracy_plot.png')
        plt.close()

        self.plot_metric(history, 'loss')
        plt.savefig('models/loss_plot.png')
        plt.close()

        # Save model summary
        with open('models/model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        print(f"Plots and summary saved to models/accuracy_plot.png, models/loss_plot.png, and models/model_summary.txt")

    def load_and_predict(self, image_path, model_path='models/image_model.h5'):
        """
        Load the image classification model from the specified path
        and make predictions on the input image.

        Args:
            image_path (str): Path to the input image file.
            model_path (str, optional): Path to the image classification model. Defaults to 'models/image_model.h5'.

        Returns:
            str: Predicted flower name.
        """
        # Get the absolute path to the imageModel.py file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the model
        model_path = os.path.join(script_dir, '../..', model_path)
        model = load_model(model_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0

        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        print(predictions)
        flower_names = {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }
        return flower_names[int(predictions)]

    def img_input(self):
        """
        Open a file dialog to allow the user to select an image file.
        
        Returns:
            str: Path to the selected image file.
        """
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        return file_path

    def summary(self):
        """
        Display the summary of the image classification model.
        """
        self.model.summary()

    @staticmethod
    def plot_metric(history, metric):
        """
        Plot the accuracy metric from the training history.

        Args:
            history (tf.keras.callback.History): Training history.
            metric (str): Metric to plot ('accuracy' or 'loss')
        """
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history[f'val_{metric}'],
                 label=f'Validation {metric}')
        plt.title(metric)
        plt.legend()
        plt.show()

    def plot_accuracy(self, history):
        """
        Plot the accuracy metric from the training history.

        Args:
            history (tf.keras.callbacks.History): Training history.
        """
        self.plot_metric(history, 'accuracy')

    def plot_losses(self, history):
        """
        Plot the loss metric from the training history.

        Args:
            history (tf.keras.callbacks.History): Training history.
        """
        self.plot_metric(history, 'loss')
