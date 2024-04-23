import matplotlib.pyplot as plt
import tensorflow as tf

class DataHandler:
    """
    A utility class for handling image data, including loading, preprocessing,
    splitting into training, validation, and test sets, and displaying sample images.

    Attributes:
        directory (str): Directory path containing the image data.
        data (tf.data.Dataset): Loaded image dataset.

    Methods:
        _load_data: Load the image data from specified  directory.
        preprocess_data(): Preprocess the loaded image data by normalizing pixel values.
        split_data(train_size=07, val_size=02, test_size=01): Split the image dataset into training, validation, and test sets.
        display_data(num_images=4): Display a batch of images from the dataset.
    """

    def __init__(self, directory):
        """
        Initialize the DataHandler class with the directory containing the image data.

        Args:
            directory (str): Directory path containing the image data.
        """
        self.directory = directory
        self.data = self._load_data()

    def _load_data(self):
        """
        Load the image data from the directory.

        Returns:
            tf.data.Dataset: Loaded image dataset.
        """
        return tf.keras.preprocessing.image_dataset_from_directory(self.directory)

    def preprocess_data(self):
        """
        Preprocess the loaded image data by normalizing pixel values.

        Returns:
            tf.data.Dataset: Preprocessed image dataset.
        """
        processed_data = self.data.map(lambda x, y: (x / 255.0, y))
        return processed_data

    def split_data(self, train_size=0.7, val_size=0.2, test_size=0.1):
        """
        Split the image dataset into training, validation, and test sets.

        Args:
            train_size (float, optional): Fraction of the dataset to use for training. Defaults to 0.7.
            val_size (float, optional): Fraction of the dataset to use for validation. Defaults to 0.2.
            test_size (float, optional): Fraciton of the dataset to use for testing. Defaults to 0.1.

        Returns:
            tuple: Training, validation, and test datasets.
        """
        total_size = sum(1 for _ in self.data)
        train_split = int(total_size * train_size)
        val_split = int(total_size * val_size)
        test_split = int(total_size * test_size)

        train_data = self.data.take(train_split)
        val_data = self.data.skip(train_split).take(val_split)
        test_data = self.data.skip(val_split + train_split).take(test_split)

        return train_data, val_data, test_data

    def display_data(self, num_images=4):
        """
        Display a batch of images from the dataset.

        Args:
            num_images (int, optional): Number of images to display. Defaults to 4.
        """
        batch = self.data.as_numpy_iterator().next()
        images, labels = batch

        fig, ax = plt.subplots(ncols=num_images, figsize=(20, 20))
        for idx in range(num_images):
            ax[idx].imshow(images[idx].astype(int))
            ax[idx].title.set_text(labels[idx])
        plt.show()
